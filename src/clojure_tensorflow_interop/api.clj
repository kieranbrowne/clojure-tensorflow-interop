(ns clojure-tensorflow-interop.api
  (:require [clojure-tensorflow-interop.utils
             :as utils :refer [tensor->clj clj->tensor]])
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


(defn op-builder
  "Returns a function which creates an operation for the graph"
  [op-profile]
  (let [{:keys [operation node-name attrs inputs]
         :or {node-name (str (gensym "tf")) attrs {} inputs []}
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
                (map (fn [input] #(.addInput % (input graph))) inputs)
                #(.build %)
                #(.output % 0)])))))



(defn constant [val]
  (let [tensor (clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor}})))

(defn mult [a b]
  (op-builder
   {:operation "Mul"
    :inputs [a b]}))

(defn add [a b]
  (op-builder
   {:operation "Add"
    :inputs [a b]}))

(defn n-args
  "This function takes a two argument operation like mult and add and
  returns a version which can take 2 -> infinity arguments like normal
  clojure functions."
  [func]
  (fn [& args]
    (loop [inputs (drop 2 args)
           out (apply func (take 2 args))]
      (if (empty? inputs)
        out
        (recur (rest inputs) (func out (first inputs)))))))

(def * (n-args mult))
(def + (n-args add))


(apply + (into [] (take 2 '(1 2))))

(defn * [& inputs]
  (op-builder
   {:operation "Mul"
    :inputs inputs}))


;; In this file I'm experimenting with possibilities for a tensorflow api
;; which feels clojurian.

;; Design decisions
;; Almost all functions return a function the last argument of which is
;; the graph object. This allows us to hold off building the graph until
;; all operations are set up and filter the graph object down to all the
;; nodes.


(defn session-run
  ""
  [operation]
  (let [graph (new Graph)
        op (operation graph)
        copy-to (utils/output-shape op)
        ]
    ((if (utils/array? copy-to)
       #(.copyTo % (utils/output-shape op))
       #(.intValue %)
       )
    (-> (new Session graph)
        .runner
        (.fetch (.name (.op op)))
        .run
        (.get 0)
        ;; .dataType
        ;; (.copyTo (utils/output-shape op))
        ))
    ))

(def a (constant [1.]))
(def b (constant [2.]))
(def c (constant [2.]))
(def d (constant [9.]))
(def a*b (mult a b))

(time
 (session-run (* a b)))
(time
 (tensor->clj
  (session-run (* a b c))))

(tensor->clj
  (session-run (+ a b c d)))

(def a (constant 2))
(def b (constant 3))
((variable 3) (new Graph))

(map #(name (first %)) {:dtype 1 :value 3})



(defn outputify [name tensor]
  (->
   (.opBuilder graph "Const" name)
   (.setAttr "dtype" (.dataType tensor))
   (.setAttr "value" tensor)
   .build
   (.output 0)
   ))


;; potential clj-to-ops macro




;; (defn run-in-session [graph op]
;;   (utils/recursively utils/array? vec
;;                (-> (new Session graph)
;;                    .runner
;;                    (.fetch (.name (.op op)))
;;                    .run
;;                    (.get 0)
;;                    (.copyTo (output-shape op))
;;                    )))
