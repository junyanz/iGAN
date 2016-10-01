FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/gvm/datasets/$FILE
OUTPUT_FILE=./datasets/$FILE

echo "Downloading the hdf5 dataset ($FILE)..."
wget -N $URL -O $OUTPUT_FILE
