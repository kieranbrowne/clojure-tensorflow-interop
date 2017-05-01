(ns clojure-tensorflow-interop.api
  (:require [clojure-tensorflow-interop.utils
             :as utils :refer [tensor->clj clj->tensor]]
            [clojure-tensorflow-interop.api :as tf])
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


;; warning
(def default-graph (new Graph))
;; we need to do some stateful code to make this work like classic tf
;; you don't have to use it in this way though; and there are plenty of
;; benefits to writing in a more functional style.
(def global-variables (atom []))
(defn global-variables-initializer []
  @global-variables)

(defn session
  "Create a session"
  ([graph] (new Session graph))
  ([] (session default-graph)))


(defn op-builder
  "Returns a function which creates an operation for the graph"
  [op-profile]
  (let [{:keys [operation node-name attrs inputs]
         :or {node-name (str (gensym operation)) attrs {} inputs []}
         } op-profile]
    ((fn [graph]
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
                #(.output % 0)]))) tf/default-graph)))



(defn constant [val]
  (let [tensor (clj->tensor val)]
    (op-builder
     {:operation "Const"
      :attrs {:dtype (.dataType tensor)
              :value tensor
              }})))


(defn assign [var val]
  (op-builder
   {:operation "Assign"
    :inputs [var (if (utils/tf-obj? val) val (constant val))]
    }))


(defn variable
  ([val] (variable val {}))
  ([val bits]
   (let [tensor (clj->tensor val)
         var (op-builder
          (merge
           {:operation "Variable"
            :attrs {:shape (utils/tensor->shape tensor)
                    :dtype (.dataType tensor)}
            } bits))]
     (swap! global-variables conj (assign var val))
     var
    )))


;; (def var1 (variable [1] {:trainable true}))

;; (session-run (assign var1 [2]))


(defn placeholder [datatype]
  (op-builder
   {:operation "Placeholder"
    :attrs {:dtype datatype}
    }))

;; (def graph (Graph.))
;; (def sess (Session. graph))
;; (def p1 ((placeholder 1) graph))
;; (.feed (.runner sess) p1 (clj->tensor 1))
;; (class p1)
;; (op-run p1)

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

(defn sigmoid [a]
  (op-builder
   {:operation "Sigmoid"
    :inputs [a]}))

(defn gradients [x activations cost]
  (->> (map vector cost x (butlast activations))
       (map (fn [gradients [error activation]]
                 ()
                 ))
       )
  )
(op-run (gradients (tf/constant [1]) (tf/constant [2]) (tf/constant [1])))


(defn pow [a b]
  (op-builder
   {:operation "Pow"
    :inputs [a b]}))

(defn size [a]
  (op-builder
   {:operation "Size"
    :inputs [a]}))

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
    :inputs [a (constant [1 0])]}))

(with-session (tf/transpose (tf/constant [[1 1 1] [1 1 1]])))

(defn matmul [a b]
  (op-builder
   {:operation "MatMul"
    :inputs [a b]}))
;; alias
(def dot matmul)


;; (session-run (dot
;;               (constant [[1 2 3] [4 5 6]])
;;               (constant [[7 8] [9 10] [11 12]])))

;; (session-run (constant [[1 2 4] [1 2 4]]))


;; (session-run (transpose (constant [1 2])))

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

;; (tensor->clj
;;  (session-run (apply + (map constant (range 390 593)))))

;; (session-run
;;  (reduce add (map constant (range 3 3000))))

;; In this file I'm experimenting with possibilities for a tensorflow api
;; which feels clojurian.

;; Design decisions
;; Almost all functions return a function the last argument of which is
;; the graph object. This allows us to hold off building the graph until
;; all operations are set up and filter the graph object down to all the
;; nodes.

(defn feed
  "Feed value to placeholder
  Pass a map of locations to values"
  ([runner feed-map]
   (utils/thread
     runner
     (map (fn [[key val]]
            #(.feed % key val)) feed-map))))

(defn run
  [runner op]
  (.run (.fetch runner op)))

(defn ensure-graphed
  [graph ops]
  (map #(if (fn? %) (% graph) %) ops))

(defn sess-run
  "Run a list of operations on a graph"
  ([ops] (sess-run ops {}))
  ([ops feed-map]
   (let [g (Graph.) s (Session. g)] (sess-run g s ops feed-map)))

  ([graph session ops feed-map]
   (let [ops (ensure-graphed graph
              (remove
               #(contains? (set (keys feed-map)) %) ops))
         feed-map (clojure.set/rename-keys
                   feed-map
                   (zipmap (keys feed-map) (ensure-graphed graph (keys feed-map))))
         runner (.runner session)]
     (feed runner feed-map)
     (map (partial run runner) (butlast ops))
     (tensor->clj
      (.get (run runner (last ops)) 0))
     )))

;; (sess-run [x y] {x (clj->tensor 5.)})

;; (def x (partial identity))
;; (def y x)

(defn apply-graph [graph graph-def]
  (reduce
   (fn [data op]
     (if (coll? op)
       (-> data
           (update-in [:ops] into (:ops (apply-graph graph op)))
           (update-in [:graph] conj (:graph (apply-graph graph op)))
           )
       (let [graphed-op (if (fn? op) (op graph) op)]
         (-> data
             (update-in [:ops] conj op)
             (update-in [:graph] conj graphed-op)))
       ))
   {:ops #{} :graph []} graph-def))

(defn build-graph
  "Build a model"
  [g graph]
  ((apply
   (first graph)
   (map
    #(if (coll? %) (build-graph g %) %)
    (rest graph))) g))

;; (build-graph [+ 1 [+ 2 [- 1 0 9] 4]])

;; ((build-graph [constant 1]) g)

;; (sess-run g (Session. g)
;;           [(build-graph g [add [constant 1.] [constant 2.]])] {})



;; (:graph
;;  (apply-graph [x 1]))

;; (def g (Graph.))
;; (def sess (Session. g))
;; (def build-graph [constant 4.])
;; (identity)

;; (apply-graph g [constant 1.])

;; (def always-1 (apply-graph g (build-graph [constant four])))
;; (build-graph [constant four])

;; (sess-run [always-1])

(defn op-run
  "Call session runner on single op.
  Returns tensor object"
  ([op] (op-run tf/default-graph op))
  ([graph op] (op-run graph (Session. graph) op {}))
  ([graph session op] (op-run graph session op {}))
  ([graph session op feed-map]
  (-> session
      .runner
      (feed feed-map)
      (.fetch (.name (.op (if (fn? op) (op graph) op))))
      .run
      (.get 0)
      )))

;; (def g (Graph.))
;; (def sess (Session. g))
;; (def x (placeholder DataType/FLOAT))
;; (def x (constant 1.))
;; (def y (mult x (constant 2.)))
;; (def z (sub x (constant 3.)))
;; (sess-run g sess [x y] {x (clj->tensor 5.)})
;; (sess-run [x y])

;; (def runner (.runner sess))
;; (feed runner {x (clj->tensor 2.)})
;;  (run runner y)
;; (run runner z)
;; (tensor->clj (last (run runner ((constant 1.) g))))
;; ;; (feed g sess {x (clj->tensor 1.)})
;; (tensor->clj
;;  (sess-run g sess [y] {x (clj->tensor 8.)}))


;; (.feed (.runner sess) p1 (clj->tensor 1))

(defn session-run
  "Run list of ops, return last"
  ([ops] (session-run tf/default-graph ops))
  ([graph ops] (session-run graph (Session. graph) ops))
  ([graph session ops]
   (let [ops (flatten ops)
         op-run (partial op-run graph session)]

     ;; initialise global variables
     (map op-run @global-variables)

     ;; run first n ops to set up state
     (doseq [op (butlast ops)]
       (op-run op))

     ;; run final op and return value
     (tensor->clj
      (op-run (last ops))))))

(defn with-session [& ops]
  (session-run ops))

;; (session-run (constant [0.0]))

;; (def x (constant [1. 0.5 0]))
;; (def W (constant [1. 0.5 0]))

;; (session-run (mult x W))

(defprotocol Tensorflow
  "Helper functions for tensors"
  (session [x] "The stuff")
  (fn-2 [x y] "The stuff"))

(extend-type org.tensorflow.Output
  Tensorflow
  (session [x] (str x)))
