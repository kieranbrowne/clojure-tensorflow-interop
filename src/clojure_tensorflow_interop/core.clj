(ns clojure-tensorflow-interop.core
  (:use clojure.reflect clojure.pprint)
  (:import [org.tensorflow
            TensorFlow
            Tensor
            Session
            Output
            Operation
            OperationBuilder
            Graph
            DataType])
  ;; data.csv is only used for reading example data
  ;; not a requirement for using tensorflow
  (:require
   [clojure.data.csv :as csv]
   [clojure-tensorflow-interop.api :as tf]
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
;; But for me, is that I write clojure to get away from object
;; oriented programming.

;; We have all that we need from the Java api already. We can already
;; work with all binary operations, run sessions and even load existing
;; and/or pre-trained models from TensorFlow in any other language.

;; There's a real opportunity here to make a properly clojurian api.
;; There are a couple of things that I think make TensorFlow and Clojure
;; a great match.
;; TensorFlow's graph concept maps well to data structure
;; structure, and no programming language has a better story for working
;; with data structures than clojure.

;; The code is data

;; And then, with a short macro we can turn our
;; (clj->ops (/ [8 2] [2 2]))
;; => [4 1]

;; One exciting direction for this is the
;; Lisp's whole *code is data* thing.
;; Below, I've hashed out a proof of concept macro to turn a clojure
;; s-expression into a big hairy TensorFlow operation.

;; Right, lets actually do some machine learning
(def training-data
  ;; input => output
  [ [0. 0. 1.]   [0.]
    [0. 1. 1.]   [1.]
    [1. 0. 1.]   [1.]
    [1. 1. 1.]   [0.] ])

(def inputs (tf/constant (take-nth 2 training-data)))
(def outputs (tf/constant (take-nth 2 (rest training-data))))

(def random-synapse #(dec (rand 2)))

(def weights
  (tf/variable
   (repeatedly 3 #(repeatedly 1 random-synapse))))


(defn network [x]
  (tf/sigmoid (tf/matmul x weights)))

(defn error [network-output]
  (tf/div (tf/pow (tf/sub outputs network-output) (tf/constant 2.)) (tf/constant 2.0)))

(defn error' [network-output]
  (tf/sub network-output outputs))

(defn sigma' [x]
  (tf/mult x (tf/sub (tf/constant 1.) x)))


(tf/with-session
  (tf/global-variables-initializer)
  (network inputs)
  (error
   (network inputs))
  )


(defn deltas [network-output]
  (tf/matmul
   (tf/transpose inputs)
   (tf/mult
    (error' (network inputs))
    (sigma' (network inputs)))))


(def sess (tf/session))
(def sess-run (partial tf/session-run tf/default-graph sess))

(sess-run [(tf/initialise-global-variables)])

(sess-run [(network inputs)])

(sess-run
 [(repeat 10000
          (tf/assign weights (tf/sub weights (deltas (network inputs)))))
  (tf/div (tf/sum (tf/abs (error (network inputs)))) (tf/constant 4.))
  ])

(sess-run [(network (tf/constant [[1. 1. 1.]]))])
;; it motherflipping works.
