FILE=$1
URL=http://efrosgans.eecs.berkeley.edu/iGAN/models/theano_dcgan/$FILE.dcgan_theano
OUTPUT_FILE=./models/$FILE.dcgan_theano

echo "Downloading the dcgan_theano model ($FILE)..."
wget -N $URL -O $OUTPUT_FILE
