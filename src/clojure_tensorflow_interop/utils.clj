(ns clojure-tensorflow-interop.utils
  (:import [org.tensorflow
            Tensor
            ]))


(defn recursively
  "Apply function to all items in nested data structure if
  condition function is met."
  [apply-if-fn func data]
  (if (apply-if-fn data)
    (func (map (partial recursively apply-if-fn func) data))
    data))


(defn make-coll
  "Make a collection of x,y,z... dimensions"
  [fill & dims]
  (case (count dims)
    0 fill
    1 (repeat (first dims) fill)
    (repeat (first dims) (apply make-coll (rest dims)))
    ))

(def array?
  "Works like coll? but returns true if argument is array"
  #(= \[ (first (.getName (.getClass %)))))

(defn tf-vals [v]
  "Convert value into type acceptable to TensorFlow
  Persistent data structures become arrays
  Longs become 32bit integers
  Doubles become floats"
  (cond
    (coll? v)
    (if (coll? (first v))
      (to-array (map tf-vals v))
      (case (.getName (type (first v)))
        "java.lang.Long" (int-array v)
        "java.lang.Double" (float-array v)))
    (= (.getName (type v)) "java.lang.Long") (int v)
    (= (.getName (type v)) "java.lang.Double") (float v)
    ;; anything else
    true v))

(defn output-shape [op]
  (let [shape (.shape op)
        dims (map #(.size shape %)
                  (range (.numDimensions shape)))]
    (tf-vals
     (apply make-coll 0.0 dims))))

(def tensor->clj (partial recursively array? vec))

(def clj->tensor #(Tensor/create (tf-vals %)))

(defn thread
  "Approximately equivalent to -> macro.
  Required because -> must run at compile time"
  [val functions] (reduce #(%2 %1) val functions))

