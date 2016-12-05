FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/alexnet/caffe_reference_$FILE.pkl
OUTPUT_FILE=./models/caffe_reference_$FILE.pkl

echo "Downloading the dcgan_theano model ($FILE)..."
wget -N $URL -O $OUTPUT_FILE
