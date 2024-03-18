find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
find . | grep -E "~" | xargs rm -rf
