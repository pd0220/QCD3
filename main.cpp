// chi squared method implementation for numerical curve fitting
// icluding useful libraries and/or headers
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <math.h>
// Eigen library for linear algebra
#include <Eigen/Dense>

// ------------------------------------------------------------------------------------------------------------

// read given file
Eigen::MatrixXd readFile(std::string fileName)
{
    // string for lines
    std::string line;

    // data structure to store data
    Eigen::MatrixXd DataMat(0, 3);

    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);
    // check if open
    if (fileToRead.is_open())
    {
        // read line by line
        int i = 0;
        while (std::getline(fileToRead, line))
        {
            // using stringstream to write matrix
            std::stringstream dataStream(line);
            DataMat.conservativeResize(i + 1, 3);
            dataStream >> DataMat(i, 0) >> DataMat(i, 1) >> DataMat(i, 2);
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "Unable to open given file." << std::endl;
        std::exit(-1);
    }

    // return raw data
    return DataMat;
}

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] is datafile to fit
//       1st col --> some physical quantity (x)
//       2nd col --> data (y)
//       3rd col --> err (sigma)
// argv[2] is highest degree of basis functions
int main(int argc, char **argv)
{
    // file name
    std::string fileName = "None";
    // degree of highest basis function
    int deg = -1;
    // check for arguments
    if (argc > 2)
    {
        fileName = argv[1];
        deg = std::stoi(argv[2]);
    }

    // error check
    if (fileName == "None")
    {
        std::cout << "No file was given, or the file dose not exist or unavailable." << std::endl;
        std::exit(-1);
    }
    if (deg < 0)
    {
        std::cout << "No degree of basis functions was given." << std::endl;
        std::exit(-1);
    }

    // read file to matrix
    Eigen::MatrixXd const rawDataMat = readFile(fileName);

    // "length" of data
    int length = rawDataMat.rows();

    // containers for x, y, err
    Eigen::VectorXd xData(length);
    Eigen::VectorXd yData(length);
    Eigen::VectorXd errData(length);

    // fill containers
    for (int i = 0; i < length; i++)
    {
        xData(i) = rawDataMat(i, 0), yData(i) = rawDataMat(i, 1), errData(i) = rawDataMat(i, 2);
    }

    // matrix for linear equation system
    Eigen::MatrixXd MMat(length, deg + 1);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j <= deg; j++)
        {
            // basis function used on x divided by err
            MMat(i, j) = std::pow(xData(i), j) / (double)errData(i);
        }
    }

    // transposing the previous matrix
    Eigen::MatrixXd MMatTranspose = MMat.transpose();

    // RHS vector for linear equation system
    Eigen::VectorXd bVec(length);

    for (int i = 0; i < length; i++)
    {
        // data point y divided by err
        bVec(i) = yData(i) / errData(i);
    }

    // solving the linear equqation system and write fitted coefficients to screen
    std::cout << (MMatTranspose * MMat).fullPivLu().solve(MMatTranspose * bVec) << std::endl;
}