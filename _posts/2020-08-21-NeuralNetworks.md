---
title: "Neural Networks"
date: 2020-8-21
tags: [ML]
toc: true
Excerpt: "Neural Networks"
toc_label: "Table of Contents"
---

## Neural Network

```
(defparameter *a-good-minimum-error* 1.0e-9)
(defun transpose (matrix)
  "Transposes a matrix"
  (apply #'mapcar #'list matrix))  ;; cool, no?

(defun make-matrix (i j func)
  "Builds a matrix with i rows and j columns,
    with each element initialized by calling (func)"
  (map-m func (make-list i :initial-element (make-list j :initial-element nil))))

(defun make-random-matrix (i j val)
  "Builds a matrix with i rows and j columns,
    with each element initialized to a random
    floating-point number between -val and val"
  (make-matrix i j #'(lambda (x)
		       (declare (ignore x))  ;; quiets warnings about x not being used
		       (- (random (* 2.0 val)) val))))

(defun e (matrix i j)
  "Returns the element at row i and column j in matrix"
  ;; 1-based, not zero-based.  This is because it's traditional
  ;; for the top-left element in a matrix to be element (1,1),
  ;; NOT (0,0).  Sorry about that.  :-)
  (elt (elt matrix (1- i)) (1- j)))

(defun print-matrix (matrix)
  "Prints a matrix in a pleasing form, then returns matrix"
  (mapcar #'(lambda (vector) (format t "~%~{~8,4,,F~}" vector)) matrix) matrix)

;;; Matrix Multiplication

(defun multiply2 (matrix1 matrix2)
  "Multiplies matrix1 by matrix2
    -- don't use this, use multiply instead"
  (verify-multiplicable matrix1 matrix2)
  (let ((tmatrix2 (transpose matrix2)))
    (mapcar #'(lambda (vector1)
		(mapcar #'(lambda (vector2)
	       (apply #'+ (mapcar #'* vector1 vector2))) tmatrix2)) matrix1)))  ;; pretty :-)
(defun scale-list (lis)
  "Scales a list so the minimum value is 0.1 and the maximum value is 0.9.  Don't use this function, it's just used by scale-datum."
  (let ((min (reduce #'min lis))
	(max (reduce #'max lis)))
    (mapcar (lambda (elt) (+ 0.1 (* 0.8 (/ (- elt min) (- max min)))))
	    lis)))

(defun scale-datum (lis)
  "Scales all the attributes in a list of samples of the form ((attributes) (outputs))"
  (transpose (list (transpose (mapcar #'scale-list (transpose (mapcar #'first lis))))
		   (transpose (mapcar #'scale-list (transpose (mapcar #'second lis)))))))

(defun multiply (matrix1 matrix2 &rest matrices)
  "Multiplies matrices together"
  (reduce #'multiply2 (cons matrix1 (cons matrix2 matrices))))

;;; Element-by-element operations

(defun add (matrix1 matrix2 &rest matrices)
  "Adds matrices together, returning a new matrix"
  (apply #'verify-equal 'add matrix1 matrix2 matrices)
  (apply #'map-m #'+ matrix1 matrix2 matrices))

(defun e-multiply (matrix1 matrix2 &rest matrices)
  "Multiplies corresponding elements in matrices together,
        returning a new matrix"
  (apply #'verify-equal 'e-multiply matrix1 matrix2 matrices)
  (apply #'map-m #'* matrix1 matrix2 matrices))

(defun subtract (matrix1 matrix2 &rest matrices)
  "Subtracts matrices from the first matrix, returning a new matrix."
  (let ((all (cons matrix1 (cons matrix2 matrices))))
    (apply #'verify-equal 'subtract all)
    (apply #'map-m #'- all)))

(defun scalar-add (scalar matrix)
  "Adds scalar to each element in matrix, returning a new matrix"
  (map-m #'(lambda (elt) (+ scalar elt)) matrix))

(defun scalar-multiply (scalar matrix)
  "Multiplies each element in matrix by scalar, returning a new matrix"
  (map-m #'(lambda (elt) (* scalar elt)) matrix))



(defparameter *verify* t)

;;; hmmm, openmcl keeps signalling an error of a different kind
;;; when I throw an error -- a bug in openmcl?  dunno...
(defun throw-error (str)
  (error (make-condition 'simple-error :format-control str)))

(defun verify-equal (funcname &rest matrices)
  ;; we presume they're rectangular -- else we're REALLY in trouble!
  (when *verify*
    (unless (and
	     (apply #'= (mapcar #'length matrices))
	     (apply #'= (mapcar #'length (mapcar #'first matrices))))
      (throw-error (format t "In ~s, matrix dimensions not equal: ~s"
			   funcname
			   (mapcar #'(lambda (mat) (list (length mat) 'by (length (first mat))))
				   matrices))))))

(defun verify-multiplicable (matrix1 matrix2)
  ;; we presume they're rectangular -- else we're REALLY in trouble!
  (when *verify*
    (if (/= (length (first matrix1)) (length matrix2))
	(throw-error (format t "In multiply, matrix dimensions not valid: ~s"
			     (list (list (length matrix1) 'by (length (first matrix1)))
				   (list (length matrix2) 'by (length (first matrix2)))))))))

(defun map-m (function &rest matrices)
  "Maps function over elements in matrices, returning a new matrix"
  (apply #'verify-equal 'map-m  matrices)
  (apply #'mapcar #'(lambda (&rest vectors)       ;; for each matrix...
		      (apply #'mapcar #'(lambda (&rest elts)     ;; for each vector...
					  (apply function elts))
			     vectors))
	 matrices))   ;; pretty :-)



(defparameter *nand*
  '(((0.1 0.1) (0.9))
    ((0.9 0.1) (0.9))
    ((0.1 0.9) (0.9))
    ((0.9 0.9) (0.1))))


(defun convert-data (raw-data)
  "Converts raw data into column-vector data of the form that
can be fed into NET-LEARN.  Also adds a bias unit of 0.5 to the input."
  (mapcar #'(lambda (datum)
	      (mapcar #'(lambda (vec)
			  (mapcar #'list vec))
		      (list (cons 0.5 (first datum))
			    (second datum))))
	  raw-data))
(defun subtract-from-scalar (scalar matrix)
  "Subtracts each element in the matrix from scalar, returning a new matrix"
  (map-m #'(lambda (elt) (- scalar elt)) matrix))
;;; implement activation function
(defun sigmoid (x)
  (/ 1 (1+ (exp (- x)))))
(defun net-error (output correct-output)
  "Returns (as a scalar value) the error between the output and correct vectors
 the error metric ( 1/2 SUM_i (Ci - Oi)^2 )
can be matrixified simply as:
error =  0.5 ( tr[c - o] . (c - o) ) )"
  (let ((neterr (mapcar #'-  correct-output output)))
    (* 1/2 (first (first (multiply ( list neterr) (transpose (list neterr))))))))


;;;building network using matrix manipulation
(defun net-build1 (datum num-hidden-units initial-bounds)
  	(let (l )
	  (setf l (append l (list (make-random-matrix num-hidden-units (length (first (first datum))) initial-bounds))))
	  (setf l (append l (list  (make-random-matrix (length (second (first datum))) num-hidden-units initial-bounds))))))



;;;feed forward
(defun forward-propagate (datum vw)
(if vw
  (list datum (forward-propagate  (map-m #'sigmoid (multiply (pop vw) datum)) vw))
  datum))

;;; back-propogate
(defun back-propagate (datum alpha vw o/p)
(let (odelta hdelta)
  (let (forwardop l)
    (setf forwardop (forward-propagate datum vw))
  (let ((o (second (second forwardop)))(h (first (second forwardop)))
  (i (first forwardop ))
      (c o/p)
  (V (first vw))
      (W (second vw)))
 	(setf odelta (e-multiply (e-multiply (subtract c o) o) (subtract-from-scalar 1 o)))

      	(setf hdelta (e-multiply (e-multiply h (subtract-from-scalar 1 h)) (multiply (transpose W) odelta)))

       	(setf W (add W (scalar-multiply alpha (multiply odelta (transpose h)))))

       	(setf V (add V (scalar-multiply alpha (multiply hdelta (transpose i)))))

			(setf l (list V W))))))
(defun shuffle (lis)
  "Shuffles a list.  Non-destructive.  O(length lis), so
pretty efficient.  Returns the shuffled version of the list."
  (let ((vec (apply #'vector lis)) bag (len (length lis)))
    (dotimes (x len)
      (let ((i (random (- len x))))
	(rotatef (svref vec i) (svref vec (- len x 1)))
	(push (svref vec (- len x 1)) bag)))
    bag))   ;; 65 s-expressions, by the way


(defun simple-generalization (training-set testing-set num-hidden-units alpha initial-bounds max-iterations)
	(let ((iteration-num 0) (worsterror 0) (total-error 0) (layers (net-build1 training-set num-hidden-units initial-bounds)))
		(loop while (and (< iteration-num max-iterations) (>  *a-good-minimum-error*  worsterror) ) do(progn
			(setf worsterror 0)
			(setf total-error 0)
			(incf iteration-num)
			(shuffle training-set)

			;;train on half the data
			 (loop for a from 0 to (- (length training-set) 1) do(progn
				 (let ( (loutputs (forward-propagate (first (nth a training-set )) layers )))
					;;check worst error
					(if (> (net-error (first (second (second loutputs))) (first (second (nth a training-set)))) worsterror)
					    (setf worsterror (net-error (first (second (second loutputs))) (first (second (nth a training-set)))))
					   )


					(setf layers (back-propagate (first (nth a training-set ))alpha layers (second (nth a training-set))))
					(setf total-error (+ total-error (net-error (first (second (second loutputs))) (first (second (nth a testing-set))))))
				)))))
		(/ total-error (length testing-set))))



(defparameter *set* *voting-records*)

(defun net-build (datum num-hidden-units alpha initial-bounds max-iterations  &optional print-all-errors)
	(let (l )
	  (setf l (append l (list (make-random-matrix num-hidden-units (length (first (first datum))) initial-bounds))))
	  (setf l (append l (list  (make-random-matrix (length (second (first datum))) num-hidden-units initial-bounds))))
	(let ((total-error 0) )
		(loop for i from 1 to max-iterations do(progn
			(setf total-error 0)
			(shuffle datum)

			 (loop for a from 1 to (- (length datum) 1) do(progn
				 (let ((loutputs (forward-propagate (first (nth a  datum)) l )))

				   (setf l (back-propagate (first (nth a datum )) alpha l (second (nth a datum))))
				   (if (equal print-all-errors t) (print (net-error (first (second (second loutputs))) (first (second (nth a datum))))))
					(setf total-error (+ total-error (net-error (first (second (second loutputs))) (first (second (nth a datum))))))
;;		(print	(first (second (second l-outputs))))))
				)))))
		(/ total-error (length datum)))))



(defparameter *xor*
  '(((0.1 0.1) (0.1))
    ((0.9 0.1) (0.9))
    ((0.1 0.9) (0.9))
    ((0.9 0.9) (0.1))))


(defun test ()
	(print "xor average training error:")
(print (net-build (convert-data *xor*) 3 .2 5 10000 t))
	(setf *set* *voting-records*)
	(setf *set* (scale-datum *set*))
	(print "simple generalization for voting-records average error:")
(print (simple-generalization (subseq (convert-data *set*) 0 (- (floor (length *set*) 2.0) 1)) (subseq (convert-data *set*) (floor (length *set*) 2.0) (- (length *set*) 1)) 6 .1 5 10000))

(setf *set* *wine*)
	(setf *set* (scale-datum *set*))
	(print "simple generalization on testing set for wine average error ")
	(print (simple-generalization (subseq (convert-data *set*) 0 (- (floor (length *set*) 2.0) 1)) (subseq (convert-data *set*) (floor (length *set*) 2.0) (- (length *set*) 1)) 6 .1 5 10000))


	(setf *set* *mpg*)
	(setf *set* (scale-datum *set*))
	(print "simple generalization on testing set for mpg average error ")
	(print (simple-generalization (subseq (convert-data *set*) 0 (- (floor (length *set*) 2.0) 1)) (subseq (convert-data *set*) (floor (length *set*) 2.0) (- (length *set*) 1)) 6 .1 5 10000)))
```
