FILE=$1
URL=http://efrosgans.eecs.berkeley.edu/iGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
HDF5_FILE=./datasets/$FILE.hdf5
echo "Downloading the hdf5 dataset ($FILE)..."
wget -N $URL -O $ZIP_FILE
echo "Unzipping..."
unzip $ZIP_FILE -d ./datasets/
echo "Delete zip file..."
rm -rf $ZIP_FILE
