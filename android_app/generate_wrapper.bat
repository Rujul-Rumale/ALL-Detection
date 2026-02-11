@echo off
cd /d "%~dp0"
if not exist "gradle\wrapper\gradle-wrapper.jar" (
    echo Generating Gradle wrapper...
    gradle wrapper --gradle-version 8.13
) else (
    echo Gradle wrapper already exists
)
