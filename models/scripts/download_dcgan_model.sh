FILE=$1
URL=https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/$FILE.dcgan_theano
OUTPUT_FILE=./models/$FILE.dcgan_theano

echo "Downloading the dcgan_theano model ($FILE)..."
wget -N $URL -O $OUTPUT_FILE
