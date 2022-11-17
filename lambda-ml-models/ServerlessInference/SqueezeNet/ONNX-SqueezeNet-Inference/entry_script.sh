#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  # exec /usr/local/bin/aws-lambda-rie /usr/local/bin/python -m awslambdaric $@
 exec /bin/bash
else
  exec /usr/local/bin/python -m awslambdaric $@
fi     