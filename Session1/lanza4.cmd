@echo off

echo Trying SVM configuration...

REM Altura centradas:
cd "C:\Users\Xian\Documents\MCV\M3_MLCV\proxecto\M3-project\Session1"
python try_SVM.py res4.txt rbf 1 5

pause
exit