(ns clojure-tensorflow-interop.core
  (:import [org.tensorflow
            TensorFlow
            Tensor
            Session
            Output
            Operation
            OperationBuilder
            Graph
            DataType])
  (:require
   [clojure-tensorflow-interop.helpers :as tf]
   [clojure-tensorflow-interop.utils :as utils
    :refer [tensor->clj clj->tensor]]))

;; We can test our installation by running the version method
;; on the TensorFlow class.
(. TensorFlow version)
;; => "1.x.x-rc2"

;; Before we get started with the actual code, there are a few concepts
;; I need to explain otherwise none of this is going to make sense.

;; The main object of computation in TensorFlow is the *tensor*.
;; A Tensor is just a typed multi-dimensional array. Nothing scary here.

;; When we write code for TensorFlow, we're not actually running
;; computations. Instead we're composing a data structure which
;; describes the flow of data. In TensorFlow this is called a *graph*.
;; The graph will describe the flow of our data through a series of
;; operations (*ops* for short). Nothing will actually be computed
;; until we launch our graph in a *session*. The session handles the
;; execution of our graph on the CPU or GPU and returns the resulting
;; tensors.

;; In short, our clojure code will assemble a graph and fire off commands
;; to the C code using a Session object.
;; Let's get started.

;; For demonstration purposes I'm going to do the first example without
;; abstracting my code. It ain't pretty, but it should make the process
;; clearer.

;; First we need to initialise a new Graph object
(def graph (new Graph))

;; Next we're going to need some example tensors to work with.
;; Because the computation isn't running in clojure we can't just define
;; our values. Instead we're defining an operation node in the graph
;; that generates a constant.
;; First I'm creating a tensor object using class' the create method.
;; Because we're interopping with the java class we first need to turn
;; our clojure persistant vector into an array of 32bit Integers.
;; Using the arrow macro for clarity; we call the .opBuilder method on
;; our graph, the first argument is the binary operation which will run
;; in this case, its "Const". This is one of a set of possible binary ops
;; that have been implemented in native code.
;; The second argument is a name for the operation. I went with
;; "tensor-1" for simplicity, but "Joaquin Phoenix" would
;; have also worked. The only requirement is that it is unique to
;; the graph. Next we set the value and datatype attributes
;; that are required for the Const operation. Finally we build our
;; operation based on the attributes and use the output method to return
;; it. It is this returned operation that gets saved in clojure.

(def tensor-1
  (let [tensor
        (Tensor/create
         (int-array
          [360 909 216 108 777 132 256 174 999 228 324 800 264]
          ))]
    (-> graph
     (.opBuilder "Const" "tensor-1")
     (.setAttr "dtype" (.dataType tensor))
     (.setAttr "value" tensor)
     .build
     (.output 0))))

(def tensor-2
  (let [tensor
        (Tensor/create
         (int-array [5 9 2 1 7 3 8 2 9 2 3 8 8]))]
    (-> graph
        (.opBuilder "Const" "tensor-2")
        (.setAttr "dtype" (.dataType tensor))
        (.setAttr "value" tensor)
        .build
        (.output 0))))

;; Now lets add a more exciting operation to our graph.
;; Again we will call the .opBuilder method on our graph object.
;; I'm going to use the "Div" (division) operation this time.
;; Next we call the .addInput method to add our two example tensors
;; as input to the operation.
;; Again we build and output our operation object, saving it as "divide".
(def divide
  (->
   (.opBuilder graph "Div" "my-dividing-operation")
   (.addInput tensor-1)
   (.addInput tensor-2)
   .build
   (.output 0)
   ))


;; To run our newly built operations, we need to create a session object
;; based on our graph.
(def session (new Session graph))


;; We'll call the .runner method on our session to get the engine running.
;; We use the .fetch method to retrieve the divide operation by name;
;; in this case we want to pass it the name we gave to the divide
;; operation just before ("my-dividing-operation"). The .get method
;; gets our result from the returned array, this gives us a Tensor object
;; which has all the data but cannot be read easily, so finally to
;; read our results, we call the .copyTo method on the Tensor to
;; copy the contents to an integer array.
(def result
  (-> session
      .runner
      (.fetch "my-dividing-operation")
      .run
      (.get 0)
      (.copyTo (int-array 13))
      ))

;; Finally we can read our results.
(apply str (map char result))
;; => "Hello, World!"

;; So we successfully ran a basic TensorFlow graph, but that code made my
;; eyes bleed. This is partially because the TensorFlow Java api is so
;; new and doesn't have the multitudes of helper functions that python
;; has yet.

;; TensorFlow's Java API is still extremely barebones and isn't covered by
;; the TensorFlow API stability guarantees. That's likely why we don't yet
;; have a clojure TensorFlow api, although there's at least one in the
;; works.

;; We have all that we need from the Java api already. We can already
;; work with all binary operations, run sessions and even load existing
;; and/or pre-trained models from TensorFlow in any other language.

;; There's a real opportunity here to make a properly clojurian api.
;; There are a couple of things that I think make TensorFlow and Clojure
;; a great match.
;; TensorFlow's graph concept maps well to data structure
;; structure, and no programming language has a better story for working
;; with data structures than clojure.

;; Right, lets actually do some machine learning
;; For simplicity's sake, I'm going to write a very shallow neural network.
;; From here on, I'm going to start using a very light layer on interop
;; that I defined in `helpers.clj`.

;; First, we'll need some training data.
(def training-data
  ;; input => output
  [ [0. 0. 1.]   [0.]
    [0. 1. 1.]   [1.]
    [1. 1. 1.]   [1.]
    [1. 0. 1.]   [0.] ])

;; We can split out training data into inputs and outputs like so.
;; Note the use of tf/constant. This simply wraps the operationBuilder
;; and takes care of adding the Const operation to the default graph.
(def inputs (tf/constant (take-nth 2 training-data)))
(def outputs (tf/constant (take-nth 2 (rest training-data))))

;; We want to initialise our weights as a random value between -1 and 1.
;; Here we use tf/variable which creates a variable node on the graph.
(def weights
  (tf/variable
   (repeatedly 3 (fn [] (repeatedly 1 #(dec (rand 2)))))))

;; Even though we're defining nodes for the tf graph, we can still define
;; our code as functions. This is particularly nice because we can still
;; use a higher order functions like comp and partial in our code.
(defn network [x]
  (tf/sigmoid (tf/matmul x weights)))

;; For our network to learn we need to measure the difference between
;; the training outputs and our network's outputs.
(defn error [network-output]
  (tf/div (tf/pow (tf/sub outputs network-output) (tf/constant 2.)) (tf/constant 2.0)))

;; For back propagation, we need the derivative of our error and sigmoid
;; functions.
(defn error' [network-output]
  (tf/sub network-output outputs))

(defn sigmoid' [x]
  (tf/mult x (tf/sub (tf/constant 1.) x)))

(defn deltas [network-output]
  (tf/matmul
   (tf/transpose inputs)
   (tf/mult
    (error' (network inputs))
    (sigmoid' (network inputs)))))

(def train-network
  (tf/assign weights (tf/sub weights (deltas (network inputs)))))

;; So far we seem to have used a whole bunch of functions to build our
;; operations. But really we've only been using one.
;; The function `op-builder` which is defined in `helpers.clj` simply
;; wraps up a bit of object-oriented code from the java api which adds
;; operations to the graph. All the other operations we have used, just
;; pass arguments to `op-builder`. This is why we can safely wrap so much
;; functionality without worrying that the Java api will change on us.

;; The other thing that our `helpers.clj` file defines is a couple of
;; functions to make running operations a bit easier.

;; Running TensorFlow code on the CPU or GPU

;; Pattern 1: op-run

;; For running a single operation (though this can be nested) we have the
;; op-run function.
;; This will return a Tensor object.
(tf/op-run (tf/sub (tf/constant [1.]) (tf/constant [3.])))
;; => #object[org.tensorflow.Tensor 0x4cab913d "FLOAT tensor with shape [1]"]
;; To read this value back into clojure values we can use tensor->clj
(tensor->clj
 (tf/op-run (tf/sub (tf/constant [1.]) (tf/constant [3.]))))
;; => [-2.0]

;; Pattern 2: session-run

;; Converting data from TensorFlow back into clojure is important for our
;; results, but its a bottleneck in our model, where as much as possible
;; we should try to keep computation running on the CPU or GPU.
;; For this we have the tf/session-run function which takes a list of
;; operations, which can change the state of the model (such as when
;; we are training) and the takes care of converting the final result back
;; to clojure values so we can read it in the repl.
(tf/session-run
 [(tf/global-variables-initializer)
  (network inputs)])
;; This returns the results of the network before training.
;; Note also the use of tf/global-variables-initializer; this is needed
;; when we are using one or more variables in our graph. There are other
;; ways of approaching the variable initialisation problem for TensorFlow
;; graphs, but for now I've just gone with the standard solution from
;; the main TF api. Note that despite the "global" in the function name
;; this is more of a naming convention. The variable initialisation is
;; scoped to the tf/session run function and won't affect other sessions.


;; Pattern 3: global session object

;; Patterns 1 and 2 are great for testing small parts of your graph or
;; a couple of operations here and there. But when we train our network
;; we want it's trained weights to be preserved so we can actually use
;; the trained network to get shit done.

;; For this we want to create a session object at the global level.
(def sess (tf/session))
;; We also might want to make a partial of the session-run function
;; to get the best of pattern 2 as well.
(def sess-run (partial tf/session-run tf/default-graph sess))

;; Now we can break up our operations steps into logical breaks
;; initialise variables and run the untrained network
(sess-run [(tf/initialise-global-variables)
           (network inputs)])

;; Run the train-network operation 10000 times and then check the error.
(sess-run
 [(repeat 10000 train-network)
  (tf/mean (error (network inputs)))])

;; Run the network on a new example
(sess-run [(network (tf/constant [[1. 1. 1.]]))])
;; => [[0.99740285]]

;; And that's about it.
;; We've converted our eyesore object-oriented interop code to something
;; perfectly readable with just a couple of functions. The code base is
;; tiny enough to allow immediate changes if the Java api changes on us
;; and the system is flexible enough that we don't need to wait for the
;; Java api to get fleshed out to jump in and get our hands dirty.
