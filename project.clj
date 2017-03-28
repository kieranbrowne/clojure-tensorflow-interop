(defproject clojure-tensorflow-interop "0.1.0-SNAPSHOT"
  :description "How to run TensorFlow native library in Clojure"
  :url "http://kieranbrowne.com/clojure-tensorflow-interop"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.tensorflow/libtensorflow "1.0.1"
                  :native-prefix ""]]
  :native-path "/users/kieran/Downloads/jni"
  )
