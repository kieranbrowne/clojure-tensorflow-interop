# How to run TensorFlow in Clojure 

TensorFlow now has a (very alpha) Java api which means clojure gets one for free. For now, Java's api is very sparse but don’t let that stop you getting your hands dirty, it already provides everything we need to work with TensorFlow in Clojure. With just java interop and a couple of helper functions we can start writing great idiomatic Clojure. 

To get started, read [Running TensorFlow in Clojure](http://kieranbrowne.com/research/clojure-tensorflow-interop) which explains the code and the concepts.

You can also read more about TensorFlow's java api [here](https://www.tensorflow.org/versions/master/install/install_java).

## Method 1: Add the maven dependency

### Just add the dependency to `project.clj`
[org.tensorflow/tensorflow "1.1.0-rc1"]

> Note: This dependency requires java 8. If that isn't your version by default, you can force lein to use it by adding the `:java-cmd "/path/to/java"` key to your `project.clj`.

## Method 2: Build from source

TensorFlow's java api is in active development with changes and improvments added every other day. By building from source, you have access to the latest changes to TensorFlow as they are added. This method of course takes much longer.

### 1. Install cli tools
1. [Install Bazel](https://www.bazel.build/versions/master/docs/install.html)
2. [Install Maven](https://maven.apache.org/install.html)
3. [Install Swig](http://www.swig.org/Doc3.0/Preface.html)

If you use homebrew, just `brew install maven bazel swig`. I also had to `brew upgrade bazel` due to compatibility issues.

### 2. Clone TensorFlow

```sh
$ git clone git@github.com:tensorflow/tensorflow.git
$ cd tensorflow
```

### 3. Configure the build

```sh
$ ./configure
```
You will be prompted with various questions about your build. If you have a CUDA graphics card definitely say yes to the gpu options.

### 4. Build TensorFlow

It's probably a good idea to put the kettle on for this one. It took about 20 minutes on my MacBook pro.

```sh
$ bazel build -c opt //tensorflow/java:pom
```

### 5. Add to your local maven repository
```sh
$ mvn install:install-file \
      -Dfile=bazel-bin/tensorflow/java/libtensorflow.jar \
      -DpomFile=bazel-bin/tensorflow/java/pom.xml
```
