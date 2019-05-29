# Compile Probobuf
# Input Params:
#   1: compile command e.g. protoc / /usr/local/bin/protoc
#   2: source directory
#   3: target directory (for cmake delta compilation)
#   4: built cache files (for protobuf recompile)

# mkdir target directory
mkdir -p $3
mkdir -p $4

# run compile command (compile to
# echo "$1 -I=$2 --cpp_out=$4 $2/*"
$1 -I=$2 --cpp_out=$4 $2/*

# copy files changed from $4 to $3
for fullpath in $4/*; do
    f=$(echo $fullpath | sed "s/.*\///")
    rsync --checksum $4/$f $3/$f
done