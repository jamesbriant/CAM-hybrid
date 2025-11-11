#!/bin/bash

# Get
echo | openssl s_client -showcerts -servername svn-ccsm-models.cgd.ucar.edu -connect svn-ccsm-models.cgd.ucar.edu:443 2>/dev/null | openssl x509 -inform pem -out svn-ccsm-cert.crt

# Verify
openssl x509 -in svn-ccsm-cert.crt -text -noout

# Get Intermediate
wget -O incommon.crt "https://spaces.at.internet2.edu/download/attachments/24576265/InCommon%20RSA%20Server%20CA%202.pem?api=v2"



