cd external/ldif/gaps
make mesa -j

cd ../../mesh_fusion/libfusiongpu
mkdir build
cd build
cmake ..
make -j
cd ..
python setup.py build_ext -i -f
cd ../librender
python setup.py build_ext -i -f
cd ../libmcubes
python setup.py build_ext -i -f

cd ../../ldif/ldif2mesh
bash build.sh

cd ../../..

