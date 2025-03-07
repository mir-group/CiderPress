if [ "$RUNNER_OS" == "Linux" ]; then
    export os='linux'
elif [ "$RUNNER_OS" == "macOS" ]; then
    export os='macos'
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
