// chi squared method implementation for numerical curve fitting with estimating error via jackknife samples
// icluding useful libraries and/or headers
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <math.h>
#include <numeric>
// Eigen library for linear algebra
#include <Eigen/Dense>

// ------------------------------------------------------------------------------------------------------------

// read given file
// expected structure: x | y | y_err | y_jck...
Eigen::MatrixXd readFile(std::string fileName)
{
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns (3 + N_jck)
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);
    // number of columns in given file
    int numOfCols = 0;
    std::string temp;
    // count number of writes to temporary string container
    while (firstLineStream >> temp)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for lines
    std::string line;

    // data structure to store data
    Eigen::MatrixXd DataMat(0, numOfCols);

    // reopen file
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
            DataMat.conservativeResize(i + 1, numOfCols);
            for (int j = 0; j < numOfCols; j++)
            {
                dataStream >> DataMat(i, j);
            }
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

// RHS vector for linear equation system
Eigen::VectorXd RHS(int length, Eigen::VectorXd yData, Eigen::VectorXd errData)
{
    Eigen::VectorXd bVec(length);

    for (int i = 0; i < length; i++)
    {
        bVec(i) = yData(i) / errData(i);
    }

    return bVec;
}

// ------------------------------------------------------------------------------------------------------------

Eigen::VectorXd JackknifeErrorEstimation(Eigen::VectorXd coeffs, std::vector<Eigen::VectorXd> JCKCoeffs)
{
    int length = coeffs.size();
    int N = static_cast<int>(JCKCoeffs.size());
    double preFactor = (double)(N - 1) / N;
    auto sq = [](double x) { return x * x; };

    Eigen::VectorXd sigmaSqVec(length);

    for (int i = 0; i < length; i++)
    {
        sigmaSqVec(i) = 0;
        for (int j = 0; j < N; j++)
        {
            sigmaSqVec(i) += sq(coeffs(i) - JCKCoeffs[j](i));
        }
        sigmaSqVec(i) = std::sqrt(preFactor * sigmaSqVec(i));
    }

    return sigmaSqVec;
}

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] is datafile to fit
//       1st col --> some physical quantity (x)
//       2nd col --> data (y)
//       3rd col --> err (sigma)
//  rest of cols --> Jackknife samples (y_jck)
// rest of argv --> if given degree should be fitted
int main(int argc, char **argv)
{
    // file name
    std::string fileName = "None";
    // check for arguments
    fileName = argv[1];
    // container for polynomial degress
    std::vector<bool> degContainer(argc - 2, 0);
    int numOfBasis = 0;
    for (int i = 0; i < argc - 2; i++)
    {
        degContainer[i] = std::stoi(argv[i + 2]);
        if (degContainer[i])
        {
            numOfBasis++;
        }
    }

    // error check
    if (fileName == "None")
    {
        std::cout << "No file was given, or the file dose not exist or unavailable." << std::endl;
        std::exit(-1);
    }
    if (argc < 3)
    {
        std::cout << "No polynomial degrees were given." << std::endl;
        std::exit(-1);
    }

    // read file to matrix
    Eigen::MatrixXd const rawDataMat = readFile(fileName);

    // "length" of data
    int length = rawDataMat.rows();

    // number if jackknife samples
    int numJCK = rawDataMat.cols() - 3;

    // containers for x, y, err and y_jck
    Eigen::VectorXd xData(length);
    Eigen::VectorXd yData(length);
    Eigen::VectorXd errData(length);
    std::vector<Eigen::VectorXd> jackknifeContainer(numJCK, Eigen::VectorXd(length));

    // fill containers
    for (int i = 0; i < length; i++)
    {
        xData(i) = rawDataMat(i, 0), yData(i) = rawDataMat(i, 1), errData(i) = rawDataMat(i, 2);
        for (int j = 0; j < numJCK; j++)
        {
            jackknifeContainer[j](i) = rawDataMat(i, 3 + j);
        }
    }

    // matrix for linear equation system
    Eigen::MatrixXd MMat(length, numOfBasis);
    for (int i = 0; i < length; i++)
    {
        int basis = 0;
        for (int j = 0; j < static_cast<int>(degContainer.size()); j++)
        {
            if (degContainer[j] == true)
            {
                // basis function used on x divided by err
                MMat(i, basis) = std::pow(xData(i), j) / (double)errData(i);
                basis++;
            }
        }
    }

    // transposing the previous matrix
    Eigen::MatrixXd MMatTranspose = MMat.transpose();
    // matrix for the equation system
    Eigen::MatrixXd MTxM = MMatTranspose * MMat;

    // RHS vector for linear equation system
    Eigen::VectorXd bVec = RHS(length, yData, errData);

    // solving the linear equqation system for fitted coefficients
    Eigen::VectorXd coeffVector = (MTxM).fullPivLu().solve(MMatTranspose * bVec);

    // coefficients for jackknife samples
    std::vector<Eigen::VectorXd> coeffJackknife(numJCK, Eigen::VectorXd(numOfBasis));

    // making fit for jackknife samples
    for (int i = 0; i < numJCK; i++)
    {
        coeffJackknife[i] = (MTxM).fullPivLu().solve(MMatTranspose * RHS(length, jackknifeContainer[i], errData));
    }

    // error estimation
    Eigen::VectorXd errorVec = JackknifeErrorEstimation(coeffVector, coeffJackknife);

    // to screen
    std::cout << coeffVector << std::endl;
    std::cout << errorVec << std::endl;
}