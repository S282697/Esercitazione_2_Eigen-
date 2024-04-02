#include <fstream>
#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

void Solver(const MatrixXd& A, const VectorXd& b,
            const VectorXd& solution,
            double& errRelPALU, double& errRelQR);

VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b);
VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b);

int main(){

    Vector2d solution(-1.0000e+0, -1.0000e+00);
    cout << scientific << setprecision(10);

    Matrix2d A1 = Matrix2d::Zero();
    Vector2d b1 = Vector2d::Zero();

    A1.row(0)<< 5.547001962252291e-01, -3.770900990025203e-02;
    A1.row(1)<< 8.320502943378437e-01, -9.992887623566787e-01;
    b1<< -5.169911863249772e-01, 1.672384680188350e-01;

    double errRel1PALU, errRel1QR;
    Solver(A1, b1, solution, errRel1PALU, errRel1QR);

    if (errRel1PALU < 1e-15 && errRel1QR < 1e-15)
        cout << "1 - "<< "PALU: "<< errRel1PALU<< " QR: "<< errRel1QR<< endl;
    else
        cerr<< "1 - Wrong system solution found"<< endl;

    Matrix2d A2 = Matrix2d::Zero();
    Vector2d b2 = Vector2d::Zero();

    A2.row(0)<< 5.547001962252291e-01, -5.540607316466765e-01;
    A2.row(1)<< 8.320502943378437e-01, -8.324762492991313e-01;
    b2<< -6.394645785530173e-04, 4.259549612877223e-04;

    double errRel2PALU, errRel2QR;
    Solver(A2, b2, solution, errRel2PALU, errRel2QR);

    if (errRel2PALU < 1e-12 && errRel2QR < 1e-12)
        cout<< "2 - "<< "PALU: "<< errRel2PALU<< " QR: "<< errRel2QR<< endl;
    else
        cerr<< "2 - Wrong system solution found"<< endl;

    Matrix2d A3 = Matrix2d::Zero();
    Vector2d b3 = Vector2d::Zero();

    A3.row(0)<< 5.547001962252291e-01, -5.547001955851905e-01;
    A3.row(1)<< 8.320502943378437e-01, -8.320502947645361e-01;
    b3<< -6.400391328043042e-10, 4.266924591433963e-10;

    double errRel3PALU, errRel3QR;

    Solver(A3, b3, solution, errRel3PALU, errRel3QR);

    if (errRel3PALU < 1e-5 && errRel3QR < 1e-5)
        cout<< "3 - "<< "PALU: "<< errRel3PALU<< " QR: "<< errRel3QR<< endl;
    else
        cerr<< "3 - Wrong system solution found"<< endl;

    return 0;
}

void Solver(const MatrixXd& A, const VectorXd& b, const VectorXd& solution,
            double& errRelPALU, double& errRelQR) {

    VectorXd x = SolveSystemPALU(A,b);
    errRelPALU = (solution - x).norm() / solution.norm();
    x = SolveSystemQR(A,b);
    errRelQR = (solution - x).norm() / solution.norm();
}

VectorXd SolveSystemPALU(const MatrixXd& A, const VectorXd& b) {
    VectorXd x = A.fullPivLu().solve(b);
    return x;
}

VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b) {
    VectorXd x = A.householderQr().solve(b);
    return x;
}

