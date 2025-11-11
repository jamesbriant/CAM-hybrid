module cam_gp
    use ftorch
    use physics_types,  only: physics_state
    use spmd_utils,     only: masterproc
    implicit none

    public :: torch_inference

    ! Declare the torch models
    type(torch_model), save :: temp_model
    type(torch_model), save :: hum_model
    logical, save :: models_initialized = .false.

contains

    subroutine init_models(t_model, qv_model)
        ! Initialize both models
        type(torch_model), intent(inout) :: t_model, qv_model
        call torch_model_load(t_model, "/models/independent_multitask_temp_0.0.pt", torch_kCUDA)
        call torch_model_load(qv_model, "/models/independent_multitask_qv_300.0.pt", torch_kCUDA)
    end subroutine init_models

    subroutine generate_gaussian_random(array, n_chunks, n_cols, n_levels)
        ! Generates an array of normally distributed random numbers using the Box-Muller transform.
        implicit none
        integer, intent(in) :: n_chunks, n_cols, n_levels
        real(8), intent(out) :: array(n_chunks, n_cols, n_levels)
        ! --- FIX END ---
        real(8) :: u1, u2, r, theta
        integer :: i, j, k

        do k = 1, n_levels
            do j = 1, n_cols
                do i = 1, n_chunks, 2 ! Process in pairs for Box-Muller
                    call random_number(u1)
                    call random_number(u2)
                    r = sqrt(-2.0_8 * log(u1))
                    theta = 2.0_8 * 3.141592653589793_8 * u2
                    array(i, j, k) = r * cos(theta)
                    if (i + 1 <= n_chunks) then
                        array(i + 1, j, k) = r * sin(theta)
                    end if
                end do
            end do
        end do
    end subroutine generate_gaussian_random

    subroutine torch_inference(phys_state)
        ! CAM Types
        type(physics_state), intent(inout) :: phys_state(:)

        ! Torch Types
        type(torch_tensor) :: temp_batch_input_tensor, hum_batch_input_tensor
        type(torch_tensor), dimension(1) :: temp_input_list, hum_input_list
        type(torch_tensor), dimension(1) :: temp_output_list, hum_output_list

        ! Host Arrays for input and output
        ! --- Double Precision (real(8)) arrays for CESM data ---
        real(8), allocatable :: host_t_array(:,:,:), host_q_array(:,:,:)
        real(8), allocatable :: host_rand_array(:,:,:)

        real(4), allocatable :: host_temp_std_array_sp(:,:)
        real(4), allocatable :: host_hum_std_array_sp(:,:)
        real(4), allocatable :: host_temp_batch_input_array_sp(:,:)
        real(4), allocatable :: host_hum_batch_input_array_sp(:,:)
        
        integer :: i, j, k, num_chunks, num_cols, num_levels, batch_size, b_idx
        ! integer, parameter :: temp_inputs = 32, hum_inputs = 15, total_inputs = 47
        ! integer, parameter :: temp_outputs = 32, hum_outputs = 15, total_outputs = 47

        integer, parameter :: temp_model_t_inputs = 32, temp_model_q_inputs = 32
        integer, parameter :: hum_model_t_inputs = 15, hum_model_q_inputs = 15
        integer, parameter :: temp_model_total_inputs = temp_model_t_inputs + temp_model_q_inputs
        integer, parameter :: hum_model_total_inputs = hum_model_t_inputs + hum_model_q_inputs
        integer, parameter :: temp_model_outputs = 32, hum_model_outputs = 15

        ! --- Model Initialization ---
        if (.not. models_initialized) then
            call init_models(temp_model, hum_model)
            models_initialized = .true.
        end if

        ! --- Get Dimensions & Prepare Host Arrays ---
        num_chunks = size(phys_state)
        num_cols = size(phys_state(1)%t, 1)
        num_levels = size(phys_state(1)%t, 2)
        batch_size = num_chunks * num_cols

        ! Allocate 3D arrays for state and random numbers
        allocate(host_t_array(num_chunks, num_cols, num_levels))
        allocate(host_q_array(num_chunks, num_cols, num_levels))
        allocate(host_rand_array(num_chunks, num_cols, num_levels))
        
        allocate(host_temp_batch_input_array_sp(batch_size, temp_model_total_inputs))
        allocate(host_hum_batch_input_array_sp(batch_size, hum_model_total_inputs))
        allocate(host_temp_std_array_sp(batch_size, temp_model_outputs))
        allocate(host_hum_std_array_sp(batch_size, hum_model_outputs))


        ! --- DEBUG: Confirm array allocations ---
        if (masterproc) then
            print*, "DEBUG: host_temp_batch_input_array_sp allocated with size: ", size(host_temp_batch_input_array_sp, 1), size(host_temp_batch_input_array_sp, 2)
            print*, "DEBUG: host_hum_batch_input_array_sp allocated with size: ", size(host_hum_batch_input_array_sp, 1), size(host_hum_batch_input_array_sp, 2)
        end if

        ! --- Generate Random Numbers on CPU using pure Fortran ---
        call generate_gaussian_random(host_rand_array, num_chunks, num_cols, num_levels)

        ! --- DEBUG: Confirm random number generation ---
        if (masterproc) then
            print*, "DEBUG: Sample random numbers:"
            do k = 1, min(3, num_levels)
                do j = 1, min(3, num_cols)
                    do i = 1, min(3, num_chunks)
                        print*, "  host_rand_array(", i, ",", j, ",", k, ") = ", host_rand_array(i, j, k)
                    end do
                end do
            end do
        end if

        ! --- Copy from CESM types to local 3D host arrays ---
        do i = 1, num_chunks
            host_t_array(i, :, :) = phys_state(i)%t
            host_q_array(i, :, :) = phys_state(i)%q(:,:,1)
        end do

        ! --- DEBUG: Confirm data copy from CESM types ---
        if (masterproc) then
            print*, "DEBUG: Sample T and Q values from host arrays:"
            do k = 1, min(3, num_levels)
                do j = 1, min(3, num_cols)
                    do i = 1, min(3, num_chunks)
                        print*, "  host_t_array(", i, ",", j, ",", k, ") = ", host_t_array(i, j, k)
                        print*, "  host_q_array(", i, ",", j, ",", k, ") = ", host_q_array(i, j, k)
                    end do
                end do
            end do
        end if

        ! --- EDIT: Reshape 3D arrays into separate 2D batch arrays on the CPU ---
        b_idx = 0
        do j = 1, num_cols
            do i = 1, num_chunks
                b_idx = b_idx + 1
                ! For temp model: all 32 T levels + all 32 Q levels
                host_temp_batch_input_array_sp(b_idx, 1:temp_model_t_inputs) = host_t_array(i, j, 1:temp_model_t_inputs)
                host_temp_batch_input_array_sp(b_idx, temp_model_t_inputs+1:temp_model_total_inputs) = host_q_array(i, j, 1:temp_model_q_inputs)
                
                ! For hum model: first 15 T levels + first 15 Q levels  
                host_hum_batch_input_array_sp(b_idx, 1:hum_model_t_inputs) = host_t_array(i, j, 1:hum_model_t_inputs)
                host_hum_batch_input_array_sp(b_idx, hum_model_t_inputs+1:hum_model_total_inputs) = host_q_array(i, j, 1:hum_model_q_inputs)
            end do
        end do


        ! --- DEBUG: Verify the batching loop index ---
        if (masterproc) then
            print*, "DEBUG: Final b_idx after loop = ", b_idx
            if (b_idx /= batch_size) then
                print*, "ERROR: Mismatch between b_idx and batch_size!"
            end if
        end if

        ! --- Create Tensors ---
        ! Create separate 2D GPU tensors from the batched arrays
        call torch_tensor_from_array(temp_batch_input_tensor, host_temp_batch_input_array_sp, torch_kCUDA)
        call torch_tensor_from_array(hum_batch_input_tensor, host_hum_batch_input_array_sp, torch_kCUDA)
        temp_input_list(1) = temp_batch_input_tensor
        hum_input_list(1) = hum_batch_input_tensor

        ! Create 2D CPU Tensors for model OUTPUT (following cam_nn.F90 pattern)
        call torch_tensor_from_array(temp_output_list(1), host_temp_std_array_sp, torch_kCPU)
        call torch_tensor_from_array(hum_output_list(1),  host_hum_std_array_sp,  torch_kCPU)

        ! --- Perform Inference ---
        ! FTorch handles the copy from GPU result to the CPU output tensor memory
        call torch_model_forward(temp_model, temp_input_list, temp_output_list)
        call torch_model_forward(hum_model,  hum_input_list, hum_output_list)

        ! --- Perform Final Perturbation Math on the CPU ---
        ! The 2D model outputs (std devs) are now available in the host arrays.
        ! We loop through the batch to apply them back to the 3D state arrays.
        b_idx = 0
        do j = 1, num_cols
            do i = 1, num_chunks
                b_idx = b_idx + 1
                ! --- FIX: Convert single-precision std_dev back to double for calculations ---
                host_t_array(i, j, 1:temp_model_outputs) = host_t_array(i, j, 1:temp_model_outputs) + &
                                         (host_rand_array(i, j, 1:temp_model_outputs) * dble(host_temp_std_array_sp(b_idx, :)))
                
                host_q_array(i, j, 1:hum_model_outputs) = host_q_array(i, j, 1:hum_model_outputs) + &
                                         (host_rand_array(i, j, 1:hum_model_outputs) * dble(host_hum_std_array_sp(b_idx, :)))
            end do
        end do

        ! --- Copy Final Modified Data from Host Arrays back to CESM types ---
        do i = 1, num_chunks
            phys_state(i)%t = host_t_array(i, :, :)
            phys_state(i)%q(:,:,1) = host_q_array(i, :, :)
        end do

        ! --- Clean Up Tensors ---
        call torch_delete(temp_batch_input_tensor)
        call torch_delete(hum_batch_input_tensor)
        call torch_delete(temp_output_list(1))
        call torch_delete(hum_output_list(1))

        ! --- Deallocate Host Arrays ---
        deallocate(host_t_array, host_q_array, host_rand_array)
        deallocate(host_temp_std_array_sp, host_hum_std_array_sp)
        deallocate(host_temp_batch_input_array_sp, host_hum_batch_input_array_sp)

    end subroutine torch_inference

end module cam_gp

