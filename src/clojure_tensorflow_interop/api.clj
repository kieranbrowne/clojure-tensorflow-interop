(ns clojure-tensorflow-interop.api
  (:require [clojure-tensorflow-interop.utils
             :as utils :refer [tensor->clj clj->tensor]])
  (:import [org.tensorflow
            TensorFlow
            Tensor
            Session
            Shape
            Output
            Operation
            OperationBuilder
            Graph
            DataType]
           ))


(defn op-builder
  "Returns a function which creates an operation for the graph"
  [op-profile]
  (let [{:keys [operation node-name attrs inputs]
         :or {node-name (str (gensym operation)) attrs {} inputs []}
         } op-profile]
    (fn [graph]
      (utils/thread graph
              (flatten
               [#(.opBuilder % operation node-name)
                ;; set attributes if any
                (map
                 (fn [attr]
                   #(.setAttr % (name (first attr)) (second attr)))
                 attrs)
                ;; add inputs if any
                (map (fn [input]
                       #(.addInput %
                                   (if (fn? input) (input graph) input)
                                   )) inputs)
                #(.build %)
                #(.output % 0)])))))



(defn constant [val]
  (let [tensor (clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor
              }})))

(defn assign [val node-name]
  (let [tensor (clj->tensor val)]
    (op-builder
     {:operation "Assign"
      :inputs [node-name tensor]
      })))

(defn variable
  ([val] (variable {}))
  ([val bits]
  #(let [tensor (clj->tensor val)]
    ((op-builder
      (merge
       {:operation "Variable"
        :attrs {:shape (utils/tensor->shape tensor)
                :dtype (.dataType tensor)
                ;; :initializer ((constant val) %)
                :initializer init
                }
        } bits)) %))))

(defn placeholder [val]
  #(let [tensor (clj->tensor val)]
     ((op-builder
       {:operation "Placeholder"
        :attrs {
                ;; :shape (utils/tensor->shape tensor)
                ;; :value tensor
                ;; :dtype (.dataType tensor)
                :dtype DataType/FLOAT
                :value (clj->tensor 1.1)
                }
        }) %)))

(defn get [val]
  #(let [tensor (clj->tensor val)]
     ((op-builder
       {:operation "get"
        :input [val]
        }) %)))


(defn mult [a b]
  (op-builder
   {:operation "Mul"
    :inputs [a b]}))

(defn div [a b]
  (op-builder
   {:operation "Div"
    :inputs [a b]}))

(defn add [a b]
  (op-builder
   {:operation "Add"
    :inputs [a b]}))

(defn sub [a b]
  (op-builder
   {:operation "Sub"
    :inputs [a b]}))

(defn sum
  ([t] (sum t (constant 0)))
  ([t dims]
   (op-builder
    {:operation "Sum"
     :inputs [t dims]})))

(defn tanh [a]
  (op-builder
   {:operation "Tanh"
    :inputs [a]}))

(defn pow [a b]
  (op-builder
   {:operation "Pow"
    :inputs [a b]}))

(defn abs [a]
  (op-builder
   {:operation "Abs"
    :inputs [a]}))

(defn mean [a]
  (op-builder
   {:operation "Mean"
    :inputs [a (constant 0)]}))

(defn transpose [a]
  (op-builder
   {:operation "Transpose"
    :inputs [a (constant [0])]}))

(defn matmul [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]}))
;; alias
(def dot matmul)


(session-run (dot
              (constant [[1 2 3] [4 5 6]])
              (constant [[7 8] [9 10] [11 12]])))

(session-run (constant [[1 2 4] [1 2 4]]))


(session-run (transpose (constant [1 2 4])))

(defn n-args
  "This function takes a two argument operation like mult and add and
  returns a version which can take 2 -> infinity arguments like normal
  clojure functions.
  TODO: Still causes stackoverflow for many args"
  [func]
  (fn [& args] (reduce func args)))

(def * (n-args mult))
(def + (n-args add))
(def - (n-args sub))

(tensor->clj
 (session-run (apply + (map constant (range 390 593)))))

(session-run
 (reduce add (map constant (range 3 3000))))

;; In this file I'm experimenting with possibilities for a tensorflow api
;; which feels clojurian.

;; Design decisions
;; Almost all functions return a function the last argument of which is
;; the graph object. This allows us to hold off building the graph until
;; all operations are set up and filter the graph object down to all the
;; nodes.


(defn op-run
  "Call session runner on single op.
  Returns tensor object"
  ([op] (op-run (Graph.) op))
  ([graph op] (op-run graph (Session. graph) op))
  ([graph session op]
  (-> session
      .runner
      (.fetch (.name (.op (op graph))))
      .run
      (.get 0)
      )))

(defn session-run
  "Run list of ops, return last"
  [& operations]
  (let [graph (new Graph)
        session (new Session graph)
        op-run (partial op-run graph session)]

    ;; run first n ops to set up state
    (doseq [op (butlast operations)]
      (op-run op))

    ;; run final op and return value
    (tensor->clj
     (op-run (last operations)))))

(session-run (constant [0.0]))

(def x (constant [1. 0.5 0]))
(def W (constant [1. 0.5 0]))

(session-run (mult x W))

