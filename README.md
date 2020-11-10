# mnist-train-save

Loads handwritten digits from [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/); trains a CNN to recognize handwritten digits in the browser; the progress is visualized with d3.

# Get it running

## Prerequisites

1. You need npm [Get npm](https://www.npmjs.com/get-npm).
2. Install packages
   ```
   npm install
   ```
3. Download the MNIST-Dataset files from [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/) and put the \*.gz files into the folder named ressources.

## Debugging

1. Compile and start webpack-dev-server
   ```
   npm start
   ```
2. navigate to http://localhost:9000/

## Build

1. Run the build script:
   ```
   npm run build
   ```
2. Take a loot at the output folder "./dist"!
