/**
 * @file matrix.h
 * @brief Header file for matrix operations.
 */

#ifndef matrix_H_
#define matrix_H_

#include <stddef.h>
#include <assert.h>  // for assert
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifndef matrix_MALLOC
#include <stdlib.h>  // for malloc
#define matrix_MALLOC malloc
#endif //NN_MALLOC

#ifndef matrix_ASSERT
#include <assert.h>
#define matrix_ASSERT assert
#endif // NN_ASSERT
#include <stdbool.h>

/**
 * @struct Mat
 * @brief Represents a matrix with its dimensions, stride, and elements.
 */
typedef struct {
    size_t rows;       /**< Number of rows in the matrix. */
    size_t cols;       /**< Number of columns in the matrix. */
    size_t stride;     /**< Number of elements between successive rows (useful for submatrices). */
    float *es;         /**< Pointer to the matrix elements. */
} Mat;

/**
 * @def MAT_AT(m, i, j)
 * @brief Accesses the element at the i-th row and j-th column of matrix m.
 */
#define MAT_AT(m, i , j) (m).es[(i)*(m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m, 0);

/**
 * @brief Generates a random float between 0 and 1.
 * @return Random float between 0 and 1.
 */
float rand_float(void)
{
    return (float)rand() / (float) RAND_MAX;
}


/**
 * @brief Allocates memory for a matrix with the given number of rows and columns.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return A newly allocated matrix.
 */
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (float*)matrix_MALLOC(sizeof(*m.es)*rows*cols);
    matrix_ASSERT(m.es != NULL);
    return m;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

/**
 * @brief Applies the sigmoid function element-wise to the matrix.
 * @param m Matrix to be modified in-place.
 */
void mat_sig(Mat m)
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

/**
 * @brief Performs matrix multiplication: dst = a * b.
 * @param dst Resultant matrix.
 * @param a First matrix.
 * @param b Second matrix.
 */
void mat_dot(Mat dst, Mat a, Mat b)
{
    matrix_ASSERT(a.cols == b.rows);
    matrix_ASSERT(dst.rows == a.rows);
    matrix_ASSERT(dst.cols == b.cols);
    size_t n = a.cols;
    
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < n; k++){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

/**
 * @brief Extracts a row from the matrix and returns it as a new matrix.
 * @param m Matrix from which to extract the row.
 * @param row Index of the row to be extracted.
 * @return Extracted row as a new matrix.
 */
Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}

/**
 * @brief Extracts a column from the matrix and returns it as a new matrix.
 * @param m Matrix from which to extract the column.
 * @param col Index of the column to be extracted.
 * @return Extracted column as a new matrix.
 */
Mat mat_col(Mat m, size_t col)
{
    Mat result = mat_alloc(m.rows, 1);
    for(size_t i = 0; i < m.rows; i++){
        MAT_AT(result, i, 0) = MAT_AT(m, i, col);
    }
    return result;
}

/**
 * @brief Copies the content of one matrix to another.
 * @param dst Destination matrix.
 * @param src Source matrix.
 */
void mat_copy(Mat dst, Mat src)
{
    matrix_ASSERT(dst.rows == src.rows);
    matrix_ASSERT(dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

/**
 * @brief Adds the elements of one matrix to another matrix.
 * @param dst Matrix to be updated.
 * @param a Matrix to be added.
 */
void mat_sum(Mat dst, Mat a)
{
    matrix_ASSERT(dst.rows == a.rows);
    matrix_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

/**
 * @brief Prints the content of the matrix.
 * @param m Matrix to be printed.
 * @param name Name of the matrix (for display purposes).
 * @param padding Number of spaces for indentation.
 */
void mat_print(Mat m, const char *name, int padding)
{
    printf("%*s%s: \n", (int)padding, "", name);
    for(size_t i = 0; i < m.rows; i++){
        printf("%*s    ", (int)padding, "");
        for(size_t j = 0; j < m.cols; j++){
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Fills the matrix with a specified value.
 * @param m Matrix to be filled.
 * @param x Value to fill the matrix with.
 */
void mat_fill(Mat m, float x)
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = x;
        }
    }
}

/**
 * @brief Fills the matrix with random values in the specified range.
 * @param m Matrix to be filled.
 * @param low Lower bound of the random values.
 * @param high Upper bound of the random values.
 */
void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = rand_float()*(high-low)+low;
        }
    }
}

/**
 * @brief Fills the matrix with a specified value in a symmetric pattern.
 * @param m Matrix to be filled.
 * @param low Lower bound of the values.
 * @param high Upper bound of the values.
 */
void mat_symmetric_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.cols; i++){
        float number = rand_float()*(high-low)+low;
        for(size_t j = 0; j < m.rows; j++){
            MAT_AT(m, j, i) = number;
        }
    }
}

/**
 * @brief Makes the matrix diagonally dominant.
 * @param m Matrix to be modified.
 */
void diagonalDominant(Mat m)
{
    for(size_t i = 0; i < m.rows; i++){
        float sum = 0;
        for(size_t j = 0; j < m.cols; j++){
            if(i != j){
                sum += fabs(MAT_AT(m, i, j));
            }
        }
        MAT_AT(m, i, i) = sum + 1;
    }
}

/**
 * @brief Fills the matrix with values from a flat data array.
 * @param m Matrix to be filled.
 * @param data Flat array of values.
 */
void fill_matrix(Mat m, float *data)
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = data[i*m.cols + j];
        }
    }
}

/**
 * @brief Fills the matrix with values from a flat array.
 * @param m Matrix to be filled.
 * @param array Flat array of values.
 */
void fill_with_array(Mat m, float array[])
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = array[i*m.cols + j];
        }
    }

}

/**
 * @brief Fills a specific column of the matrix with values from a flat array.
 * @param m Matrix to be filled.
 * @param array Flat array of values.
 * @param col Column index to be filled.
 */
void fill_with_array_col(Mat m, float array[], size_t col)
{
    for(size_t i = 0; i < m.rows; i++){
        MAT_AT(m, i, col) = array[i];
    }
}

/**
 * @brief Copies a column from one matrix to another matrix.
 * @param dst Destination matrix.
 * @param src Source matrix.
 * @param destCol Destination column index.
 * @param srcCol Source column index.
 */
void copy_col(Mat dst, Mat src, int destCol, int srcCol)
{
    for(size_t i = 0; i < src.rows; i++){
        MAT_AT(dst, i, destCol) = MAT_AT(src, i, srcCol);
    }
}

/**
 * @brief Switches two rows in the matrix.
 * @param m Matrix to be modified.
 * @param row1 Index of the first row.
 * @param row2 Index of the second row.
 */
void switchRow(Mat m, size_t row1, size_t row2) {
    for (int i = 0; i < m.cols; ++i) {
        float temp = MAT_AT(m, row1, i);
        MAT_AT(m, row1, i) = MAT_AT(m, row2, i);
        MAT_AT(m, row2, i) = temp;
    }
}

/**
 * @brief Fills the matrix with the identity matrix.
 * @param m Matrix to be filled.
 * @return Matrix filled with the identity values.
 */
Mat fill_identity(Mat m)
{
    m = mat_alloc(m.rows, m.cols);
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            if(i == j){
                MAT_AT(m, i, j) = 1;
            } else {
                MAT_AT(m, i, j) = 0;
            }
        }
    }
    return m;
}

/**
 * @brief Subtracts two columns from different matrices.
 * @param a First matrix.
 * @param b Second matrix.
 * @param colA Column index from matrix a.
 * @param colB Column index from matrix b.
 * @return Resulting matrix after subtraction.
 */
Mat subtract_cols(Mat a, Mat b, size_t colA, size_t colB)
{
    Mat result = mat_alloc(a.rows, 1);
    for(size_t i = 0; i < a.rows; i++){
        MAT_AT(result, i, 0) = MAT_AT(a, i, colA) - MAT_AT(b, i, colB);
    }
    return result;
}

/**
 * @brief Prepares the matrix for elimination during Gaussian elimination.
 * @param m Matrix to be prepared.
 * @param x Placeholder matrix.
 * @param identity Identity matrix.
 * @param index Index indicating the current step in elimination.
 * @param ndim Number of dimensions (rows/columns) in the matrix.
 */
void prepare(Mat m, Mat x, Mat identity, size_t index, size_t ndim) {
    matrix_ASSERT(m.rows == x.rows);

    if(index >= m.cols || index >= m.rows){
        return;
    }

    for (int row = index; row <= ndim; ++row) {
        bool largest = false;

        while (!largest) {
            largest = true;

            for (int i = index + 1; i <= ndim; ++i) {
                if (MAT_AT(m, row, index) < MAT_AT(m, i, index)) {
                    largest = false;
                    switchRow(m, row, i);
                    switchRow(identity, row, i);
                    switchRow(x, row, i);
                }
            }
        }
    }
}

/**
 * @brief Performs Gaussian elimination on the matrix.
 * @param m Matrix to be modified.
 * @param x Placeholder matrix.
 */
void eliminate(Mat m, Mat x)
{
    for(size_t i = 0; i < m.cols; i++){
        //prepare(m, x, identity, i, m.rows - 1);
        for(size_t j = i + 1; j < m.rows; j++){
            if(MAT_AT(m, j, i) == 0 || MAT_AT(m, i, i) == 0){
                continue;
            } 
            float ratio = MAT_AT(m, j, i) / MAT_AT(m, i, i);
            for(size_t k = i; k < m.cols; k++){
                MAT_AT(m, j, k) -= ratio * MAT_AT(m, i, k);
            }
            MAT_AT(x, j, 0) -= ratio * MAT_AT(x, i, 0);
        }
    }
}

/**
 * @brief Performs back substitution on the matrix to find the solution.
 * @param m Matrix resulting from Gaussian elimination.
 * @param x Placeholder matrix.
 * @return Solution matrix.
 */
Mat back_sub(Mat m, Mat x)
{

    matrix_ASSERT(m.rows == x.rows);
    Mat result = mat_alloc(m.rows, 1);
    mat_copy(result, x);
    for(int i = m.rows - 1; i >= 0; i--){
        for(int j = m.cols; j > (i); j--){
            MAT_AT(result, i, 0) -= MAT_AT(m, i, j) * MAT_AT(result, j, 0);
        }
        if(MAT_AT(m, i, i) != 0){
            MAT_AT(result, i, 0) /= MAT_AT(m, i, i);
        }else{
            MAT_AT(result, i, 0) = 0;
        }
    }
    return result;
}

/**
 * @brief Transposes the matrix.
 * @param m Matrix to be transposed.
 * @return Transposed matrix.
 */
Mat mat_transpose(Mat m)
{
    Mat result = mat_alloc(m.cols, m.rows);
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(result, j, i) = MAT_AT(m, i, j);
        }
    }
    return result;
}

/**
 * @brief Multiplies two matrices.
 * @param a First matrix.
 * @param b Second matrix.
 * @return Resulting matrix after multiplication.
 */
Mat mat_mul(Mat a, Mat b)
{
    matrix_ASSERT(a.cols == b.rows);
    Mat result = mat_alloc(a.rows, b.cols);
    mat_dot(result, a, b);
    return result;
}

void mat_div(Mat a, float nbr){
    for (size_t r = 0; r < a.rows; r++){
        for (size_t c = 0; c < a.cols; c++){
            MAT_AT(a, r, c) /= nbr;
        }
    }
}

/**
 * @brief Solves a system of linear equations using Gaussian elimination and back substitution.
 * @param m Coefficient matrix.
 * @param x Column matrix representing the right-hand side of the equation.
 * @return Matrix representing the solution vector.
 */
Mat solve(Mat m, Mat x)
{
    eliminate(m, x);
    return back_sub(m, x);
}

/**
 * @brief Solves a system of linear equations using the least squares method.
 * @param m Coefficient matrix.
 * @param x Column matrix representing the right-hand side of the equation.
 * @return Matrix representing the solution vector.
 */
Mat solve_leastSquares(Mat m, Mat x)
{
    Mat transposed = mat_transpose(m);
    Mat mtm = mat_alloc(transposed.rows, m.cols);
    mat_dot(mtm, transposed, m);
    Mat right = mat_alloc(transposed.rows, x.cols);
    mat_dot(right, transposed, x);
    return solve(mtm, right);
}

/**
 * @brief Prints the solution of a system of linear equations to the console.
 * @param m Coefficient matrix.
 * @param x Column matrix representing the right-hand side of the equation.
 */
void print_result(Mat m, Mat x)
{
    Mat result = solve(m, x);
    for(size_t i = 0; i < m.rows; i++){
        printf("x%zu = %f\n", i, MAT_AT(result, i, 0));
    }
}

/**
 * @brief Prints the elements of a matrix to the console.
 * @param x Matrix to be printed.
 */
void print_resultMatrix(Mat x)
{
    for(size_t i = 0; i < x.rows; i++){
        printf("x%zu = %f\n", i, MAT_AT(x, i, 0));
    }
}

/**
 * @brief Performs the Gram-Schmidt process on the given matrix to obtain an orthogonalized matrix.
 * @param m Matrix to be orthogonalized.
 * @return Matrix obtained from the Gram-Schmidt process.
 */
Mat compute_gramSchmidt(Mat m)
{

    Mat R = mat_alloc(m.cols, m.cols);
    mat_fill(R, 0);
    Mat Q = mat_alloc(m.rows, m.rows);
    mat_copy(Q, m);

    for(size_t i = 0; i < m.cols; i++){
        Mat vi = mat_col(m, i);
        float k;
        
        for(size_t j = 0; j < i; j++){
            Mat qi = mat_col(Q, j);
            Mat transpose = mat_transpose(qi);
            k = MAT_AT(mat_mul(transpose, vi), 0, 0);
            MAT_AT(R, j, i) = k;
            //MAT_AT(Q, i, 0) = (vi, qi, i, 0);
            for(size_t h = 0; h < m.rows; h++){
                MAT_AT(Q, h, i) -= MAT_AT(qi, h, 0) * k;
            }
        }

        float norm = 0;
        for(size_t j = 0; j < Q.rows; j++){
            norm += MAT_AT(Q, j, i) * MAT_AT(Q, j, i);
        }
        norm = sqrt(norm);
        MAT_AT(R, i, i) = norm;
        MAT_PRINT(R);
        if(norm == 0) continue;

        for(size_t j = 0; j < Q.rows; j++){
            MAT_AT(Q, j, i) = MAT_AT(Q, j, i) / MAT_AT(R, i, i);
        }
    }
    MAT_PRINT(Q);
    MAT_PRINT(R);
    return mat_mul(Q, R);
}

#endif //NN_H_

