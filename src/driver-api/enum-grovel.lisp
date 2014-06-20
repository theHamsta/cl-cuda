#|
  This file is a part of cl-cuda project.
  Copyright (c) 2014 Masayuki Takagi (kamonama@gmail.com)
|#


(in-package :cl-cuda.driver-api)


;;;
;;; Include CUDA header file
;;;

#+darwin (include "cuda/cuda.h")
#+linux (include "cuda.h")


;;;
;;; Enumerations
;;;

(cenum (cu-event-flags-enum :define-constants t)
  ((:cu-event-default "CU_EVENT_DEFAULT"))
  ((:cu-event-blocking-sync "CU_EVENT_BLOCKING_SYNC"))
  ((:cu-event-disable-timing "CU_EVENT_DISABLE_TIMING"))
  ((:cu-event-interprocess "CU_EVENT_INTERPROCESS")))