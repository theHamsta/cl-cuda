#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.nvcc
  (:use :cl
        :cl-cuda.api.nvrtc)
  (:export :*tmp-path*
           :*nvcc-options*
           :*nvcc-binary*
           :nvcc-compile
           :nvrtc-compile))
(in-package :cl-cuda.api.nvcc)

(defvar *has-nvrtc* nil)

(cffi:define-foreign-library libnvrtc
  (:darwin (:framework "CUDA"))
  (:unix (:or "libnvrtc.so")))
(unless *has-nvrtc*
  (handler-case (progn
                  (cffi:use-foreign-library libnvrtc)
                  (setf *has-nvrtc* t))
    (cffi:load-foreign-library-error (e)
      (princ e *error-output*)
      (terpri *error-output*))))

;; TODO (stephan): ensure thread-safely that there is not two processes writing out the same cache file

;;;
;;; Helper
;;;

(defparameter *tmp-path* (ensure-directories-exist (merge-pathnames #P".cache/cl-cuda/" (user-homedir-pathname))))
(defvar *prefer-jit-compilation* t)
(defvar *string-dump* (make-hash-table :test #'equalp))

(defun alloc-c-string (string &optional (dump *string-dump*))
  (let ((rtn (gethash string dump)))
    (unless rtn
      (setf rtn (cffi:foreign-string-alloc string))
      (setf (gethash string dump) rtn))
    rtn))

(defun get-tmp-path ()
  *tmp-path*)

(defun to-hex-string (bytes &aux (size (* 2 (length bytes))))
  (let ((hex (make-array size :element-type 'character :fill-pointer 0)))
    (prog1 hex
      (with-output-to-string (o hex)
        (map () (lambda (byte) (format o "~(~2,'0x~)" byte)) bytes)))))

(defun get-hashsum (code)
   (to-hex-string (md5:md5sum-string code)))

(defun get-cu-path (cuda-code)
  (let ((name (format nil "cl-cuda.~A" (get-hashsum cuda-code))))
    (make-pathname :name name :type "cu" :defaults (get-tmp-path))))

(defun get-ptx-path (cu-path)
  (make-pathname :type "ptx" :defaults cu-path))

(defun get-include-path ()
  (asdf:system-relative-pathname :cl-cuda #P"include/"))

(defvar *nvcc-options* nil)

(defun get-nvcc-options (cu-path ptx-path)
  (let ((include-path (get-include-path)))
    (append *nvcc-options*
            (list "-I" (namestring include-path)
                  "-ptx"
                  "-o" (namestring ptx-path)
                  (namestring cu-path)))))

;;;
;;; Compiling with invoking NVCC
;;;

(defun nvcc-compile (cuda-code)
  (let* ((cu-path (get-cu-path cuda-code))
         (ptx-path (get-ptx-path cu-path)))
    (write-file cu-path cuda-code)
    (unless (probe-file ptx-path)
      (print-nvcc-command cu-path ptx-path)
      (run-nvcc-command cu-path ptx-path))
    (namestring ptx-path)))

(defun write-file (file content)
  (with-open-file (out file :direction :output :if-exists :supersede)
    (princ content out)))

(defun defer-write-file (file content)
  (write-file file content))

(defvar *nvcc-binary* "nvcc"
  "Set this to an absolute path if your lisp doesn't search PATH.")

(defun print-nvcc-command (cu-path ptx-path)
  (let ((options (get-nvcc-options cu-path ptx-path)))
    (format t "~A~{ ~A~}~%" *nvcc-binary* options)))

(defun run-nvcc-command (cu-path ptx-path)
  (let ((options (get-nvcc-options cu-path ptx-path)))
    (with-output-to-string (out)
      (multiple-value-bind (status exit-code)
          (external-program:run *nvcc-binary* options :error out)
        (unless (and (eq status :exited) (= 0 exit-code))
          (error "nvcc exits with code: ~A~%~A" exit-code
                 (get-output-stream-string out)))))))

;;;
;;; Compiling using nvrtc
;;;
(defvar *builtin-headers*
  '(#P"int.h"
    #P"float.h"
    #P"float3.h"
    #P"float4.h"
    #P"float4.h"
    #P"double.h"
    #P"double3.h"
    #P"double4.h"))

(defvar *builtin-header-contents*
  (mapcar (lambda (file)
            (cons
              (alloc-c-string
                (alexandria:read-file-into-string
                  (merge-pathnames file (get-include-path))))
              (alloc-c-string (namestring file))))
          *builtin-headers*))

(defun nvcc-options-without-arch ()
  (append *nvcc-options* (list "-I" (namestring (get-include-path)))
  ;(remove-if-not (lambda (option) (let ((pos (search "-arch" option)))
                                   ;(or (not pos) (/= 0 pos))))
                 ;(append *nvcc-options* (list "-I" (namestring (get-include-path)))))
  ))

(defun nvrtc-compile (cuda-code)
  "Compile CUDA code using nvrtc (Nvidia's runtime compilation library).
  Falls back to nvcc if nvrtc is not available"
  (if (and *has-nvrtc* *prefer-jit-compilation*)
      (let* ((ptx-path (get-ptx-path (get-cu-path cuda-code)))
             (compile-options (nvcc-options-without-arch)))
        (if (probe-file ptx-path)
            (namestring ptx-path)
            (progn
              (cffi:with-foreign-strings ((c-cuda-code cuda-code)
                                          (c-compile-options 
                                            (apply #'concatenate `( string ,@(loop for o in compile-options
                                                                                   collect (format nil "~A~A" o #\Null))))))
                (cffi:with-foreign-objects ((program 'nvrtcProgram)
                                            (cached-headers '(:pointer :char) (length *builtin-header-contents*))
                                            (cached-header-names '(:pointer :char) (length *builtin-header-contents*))
                                            (c-options-pointer '(:pointer :char) (length compile-options))
                                            (ptx-size '(:pointer :pointer))
                                            (log-size '(:pointer :pointer)))
                  (loop for header in *builtin-header-contents*
                        for i from 0
                        do (setf (cffi:mem-aref cached-headers '(:pointer :char) i) (car header))
                        do (setf (cffi:mem-aref cached-header-names '(:pointer :char) i) (cdr header)))
                  (loop for o in compile-options
                        for i from 0
                        with offset = 0
                        do (setf (cffi:mem-aref c-options-pointer '(:pointer :char) i)
                                 (alloc-c-string o))
                        do (setf offset (+ 1 (length o))))
                  (assert (equalp :nvrtc-success (nvrtcCreateProgram program
                                                                     c-cuda-code
                                                                     (cffi:null-pointer)
                                                                     0
                                                                     (cffi:null-pointer)
                                                                     (cffi:null-pointer))))
                  ;(assert (equalp :nvrtc-success (nvrtcCreateProgram program
                                                                     ;c-cuda-code
                                                                     ;(cffi:null-pointer)
                                                                     ;(length *builtin-headers*)
                                                                     ;cached-headers
                                                                     ;cached-header-names)))
                  (let ((compilation-result (nvrtcCompileProgram (cffi:mem-ref program 'nvrtcProgram)
                                                                      (length compile-options)
                                                                      c-options-pointer)))
                    (unless (equalp compilation-result :nvrtc-success)
                      (assert (equalp :nvrtc-success (nvrtcGetProgramLogSize (cffi:mem-ref program 'nvrtcProgram)
                                                                             log-size)))
                      (error (cffi:with-foreign-pointer-as-string (log (cffi:mem-aref log-size :int64))
                             (assert (equalp :nvrtc-success (nvrtcGetProgramLog (cffi:mem-ref program 'nvrtcProgram) log))))))
                  (assert (equalp :nvrtc-success (nvrtcGetPtxSize (cffi:mem-ref program 'nvrtcProgram)
                                                                  ptx-size)))
                  (let ((ptx  (cffi:with-foreign-pointer-as-string (ptx (cffi:mem-aref ptx-size :int64))
                                (assert (equalp :nvrtc-success (nvrtcGetPtx (cffi:mem-ref program 'nvrtcProgram)
                                                                            ptx))))))
                    (nvrtcDestroyProgram program)
                    (defer-write-file ptx-path ptx)
                    (list ptx))))))))
      (nvcc-compile cuda-code)))

(defun get-nvrtc-options ()
  *nvcc-options*)

