set distutils_use_sdk=1
set mssdk=1
"c:\program files\microsoft sdks\windows\v7.0\setup\windowssdkver.exe" -q -version:v7.0
"c:\program files\microsoft sdks\windows\v7.0\bin\setenv.cmd" /x64 /release
cd \Users\Dubya\Dropbox\soft\zorro_beta
python setup.py build
python setup.py bdist_wheel
