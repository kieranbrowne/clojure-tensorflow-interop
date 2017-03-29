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
            DataType]
           ))

;; We can test our installation by running the version method
;; on the TensorFlow class.
(. TensorFlow version)
;; => "1.x.x-rc2"

;; Before we get started with the actual code, there are a few concepts
;; I need to explain otherwise none of this is going to make sense.

;; The main object of computation in TensorFlow is the *tensor*.
;; A Tensor is just a typed multi-dimensional array. Nothing scary here.

;; When we write in TensorFlow, we're not actually running computations.
;; Instead we'll represent our computations as a *graph*.
;; This graph will describe the flow of our data through a series of
;; operations (*ops* for short). Nothing will actually be computed
;; until we launch our graph in a *session*. The session handles the
;; execution of our graph on the CPU or GPU and returns the resulting
;; tensors.

;; In short, our clojure code will assemble a graph and fire off commands
;; to the C code using a Session object.
;; Let's get started.


;; First we need to initialise a new Graph object
(def graph (new Graph))

(defn outputify [name tensor]
  (->
   (.opBuilder graph "Const" name)
   (.setAttr "dtype" (.dataType tensor))
   (.setAttr "value" tensor)
   .build
   (.output 0)
   ))

;; Lets get a really basic operation running.
;; We're also going to need some example tensors to work with
(def tensor-1
  (outputify "t1"
    (Tensor/create
    (int-array [8 2]))))

(Tensor/create (to-nested-ints [1 2]))

(map type (to-nested-ints [1 2]))
(type (to-array (to-nested-ints [1 2])))




(def tensor-2
  (outputify "t2"
   (Tensor/create
    (int-array [2 2]))))


;; Now lets add an operation to our graph.
;; The .opBuilder method is key here. It adds a new operation
;; to our graph object, taking an Operation and Name as args.
;; The operation name is one of a set of possible binary ops
;; that have been implemented in native code. I'm going to use
;; "Div" (the division operation) for my example. The second
;; argument is a name for the operation; this is a string that
;; the graph uses to keep track of it's parts.
;; Following our opBuilder method, we use the .addInput method
;; to add our two example tensors as input to the operation.
;; Finally we run the build and output methods to complete the
;; operation.
(def divide
  (->
   (.opBuilder graph "Div" "my-dividing-operation")
   (.addInput tensor-1)
   (.addInput tensor-2)
   .build
   (.output 0)
   ))

;; Lastly, lets initialise a session object based on our our
;; graph object.
(def session (new Session graph))


;; Get the results
(def result
  (-> session
      .runner
      (.fetch (.name (.op divide)))
      .run
      (.get 0)
      (.copyTo (int-array 2))
      ))

;;Read results
(pprint result)
;; => prints: [4, 1]



;; It's still very early days for
;; the Java api so we don't have many of the helper functions that Python
;; users get. To make up for that, lets define a few helper functions.

;; We can create tensors using the create method from the Tensor class.
;; Because a tensor is a "typed multi-dimensional array" we can't just
;; use clojure data structures, which are actually not arrays under the
;; hood.

;; TODO

;; As a simple example of deep learning, lets train a network on the XOR
;; gate.


(def graph (new Graph))

(defn outputify [name tensor]
  (->
   (.opBuilder graph "Const" name)
   (.setAttr "dtype" (.dataType tensor))
   (.setAttr "value" tensor)
   .build
   (.output 0)
   ))

(def training-input
  (outputify "train-in"
             (Tensor/create
              (to-array (map int-array [[0 0]
                                        [1 0]
                                        [0 1]
                                        [1 1]])))))

(def training-output
  (outputify "train-out"
             (Tensor/create
              (to-array (map int-array [[0]
                                        [1]
                                        [1]
                                        [0]])))))

(def synapses-0
  (outputify "syn0"
             (Tensor/create
              (to-array (map int-array [[2 2]]))
              )))


(def divide
  (->
   (.opBuilder graph "Div" "my-dividing-operation")
   (.addInput tensor-1)
   (.addInput tensor-2)
   .build
   (.output 0)
   ))

;; activation op
(def graph (new Graph))
(def divide
  (->
   (.opBuilder graph "Tanh" "dicks")
   (.addInput tensor-1)
   .build
   (.output 0)
   ))

(def session (new Session graph))


(def result
  (-> session
      .runner
      (.fetch (.name (.op divide)))
      .run
      (.get 0)
      (.copyTo
       (make-array java.lang.Integer/TYPE 1 2))
      ))

(pprint result)
(count [[1 1] [1]])


(counted? (to-array [1 2]))
(defn counted?)
(if (try (inc (count [1])) (catch Exception e false)) true false)

(contains? #{1 2 7} 6)
;; We need a fn to turn clojure data structures to types
;; that TensorFlow accepts
(coll? 1)
(defn tf-vals [v]
  (cond
    ;; list of some kind
    ;; (contains? #{"clojure.lang.PersistentVector"
    ;;              "clojure.lang.PersistentList"}
    ;;            (.getName (type v)))
    (coll? v)
    (if (coll? (first v))
      (to-array (map tf-vals v))
      (case (.getName (type (first v)))
        "java.lang.Long" (int-array v)
        "java.lang.Double" (float-array v)))
    (= (.getName (type v)) "java.lang.Long") (int v)
    (= (.getName (type v)) "java.lang.Double") (int v)
    ;; anything else
    true v))

;; potential clj-to-ops macro
(defn fn-to-op [f]
  (case f
    * "Mult"
    / "Div"
    + "Add"
    def "Const"
    java.lang.Math/tanh "Tanh"
    f))

(. (class 1) TYPE)
(type (tf-vals 1))
(type (int-array (tf-vals [1 2])))
(type (tf-vals [[1 2]]))


(defn prep-for-tf
  "Take clojure val and return tensorflow constant (Operation)"
  [val]
  (outputify (str (gensym))
   (Tensor/create (tf-vals val))))

(outputify "ballz"
 (Tensor/create (tf-vals [[1]])))

(prep-for-tf 1)
(prep-for-tf [1])
(prep-for-tf [[1]])



(defmacro clj-to-ops
  "UNFINISHED: I'm curious whether we can turn standard
  Clojure code into TensorFlow operations."
  [sexp graph]
  (let [op (fn-to-op (first sexp))
        inputs (rest sexp)]
    (list 'let (vector 'args (vec inputs))
    (concat (list '-> (list '.opBuilder graph op (str (gensym))))
            (map #(list '.addInput
                        (list 'prep-for-tf
                              (list 'get 'args %1)))
                 (range (count inputs)))
          (list '.build '(.output 0))))))

(macroexpand-1 '(clj-to-ops (* 1 (* 1 3)) graph))
(macroexpand '(clj-to-ops (* 1 (* 1 3)) graph))
(macroexpand-1 '(clj-to-ops (+ 1 1) graph))
(macroexpand '(clj-to-ops (* 1 (* 1 3)) graph))


(clj-to-ops (+ 1 2) graph)

(map-indexed #(str %1 %2) (range 1 8))

(vector 1 2)
