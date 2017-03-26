(ns clojure-tensorflow-interop.core
  (:import [org.tensorflow
            TensorFlow
            Tensor
            DataType
            Session
            Graph]))


;; setup session
(def session (Session. (Graph.)))

;; read version number
(. TensorFlow version)
