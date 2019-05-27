# TinyInfer

## Dependency
#### Libraries
- [google protobuf](https://github.com/protocolbuffers/protobuf)
- [nlohmann json](https://github.com/nlohmann/json)
- eigen3 (auto-download on building)

Notes: 
- Please make sure your protobuf library supports proto3
- It is recommended to compile and install protobuf and nlohmann-json from source code


#### Local Test Environment
- Ubuntu 16.04
- Protobuf 3.7.1

## Usage 
### Compilation
```bash
mkdir build
cd build
cmake ..
make -j3
```

## File Structure
- train: contain python scripts for training (need tensorflow to run)
- infer:
	+ proto: protobuf message defination for tensorflow