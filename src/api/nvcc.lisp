#|
  This file is a part of cl-cuda project.
  Copyright (c) 2012 Masayuki Takagi (kamonama@gmail.com)
|#

(in-package :cl-user)
(defpackage cl-cuda.api.nvcc
  (:use :cl)
  (:export :*tmp-path*
           :*nvcc-options*
           :*nvcc-binary*
           :nvcc-compile))
(in-package :cl-cuda.api.nvcc)


;;;
;;; Helper
;;;

(defvar *tmp-path* (ensure-directories-exist (merge-pathnames #P".cache/cl-cuda/" (user-homedir-pathname))))

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
  (asdf:system-relative-pathname :cl-cuda #P"include"))

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
    (output-cuda-code cu-path cuda-code)
    (unless (probe-file ptx-path)
      (print-nvcc-command cu-path ptx-path)
      (run-nvcc-command cu-path ptx-path))
    (namestring ptx-path)))

(defun output-cuda-code (cu-path cuda-code)
  (with-open-file (out cu-path :direction :output :if-exists :supersede)
    (princ cuda-code out)))

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
