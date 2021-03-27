#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <array>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include<Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>


const double GAUSS_POINT_1 = (1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0));
const double GAUSS_POINT_2 = (1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0));
const double GAUSS_WEIGHT_0 = 128.0 / 225.0;
const double GAUSS_WEIGHT_1 = (322.0 + 13 * sqrt(70.0)) / 900.0;
const double GAUSS_WEIGHT_2 = (322.0 - 13 * sqrt(70.0)) / 900.0;


const int NUM_STEPS = 1000;
double FINAL_TIME = 30000.0;

double RHO = 1.0;


double C_v(double x, double y)
{
	if (x <= 0.2 || y <= 0.2)
	{
		return 1088 * 1250;  //Plaster
	}
	if ((x >= 0.2 && x <= 0.14 && y >= 0.2) || (y >= 0.2 && y <= 0.14 && x >= 0.2))
	{
		return 330 * 837; // Mica Insulating Powder
	}
	if ((x >= 0.14 && x <= 0.59 && y >= 0.14) || (y >= 0.14 && y <= 0.59 && x >= 0.14))
	{
		return 1980 * 732; // Fireclay Brick
	}
	else
	{
		return 1088 * 1250; // Plaster
	}
}

Eigen::MatrixXd KAPPA(double x, double y)
{
	Eigen::Matrix2d ID = Eigen::MatrixXd::Identity(2,2);
	double kappa;
	if (x <= 0.2 || y <= 0.2)
	{
		kappa =  0.431;  //Plaster
	}
	if ((x >= 0.2 && x <= 0.14 && y >= 0.2) || (y >= 0.2 && y <= 0.14 && x >= 0.2))
	{
		kappa = 0.121; // Mica Insulating Powder
	}
	if ((x >= 0.14 && x <= 0.59 && y >= 0.14) || (y >= 0.14 && y <= 0.59 && x >= 0.14))
	{
		kappa = 1.297; // Fireclay Brick
	}
	else
	{
		kappa = 0.431; // Plaster
	}

	
	return (kappa) * ID;
}


double ksi_a(int a)
{
	if (a == 1 || a == 4)
	{
		return -1.0;
	}
	if(a == 2 || a == 3)
	{
		return 1.0;
	}
	else
	{
		std::cout << "The passed index for ksi_a is not valid" << std::endl;
	}
}

double eta_a(int a)
{
	if (a == 1 || a == 2)
	{
		return -1.0;
	}
	if (a == 3 || a == 4)
	{
		return 1.0;
	}
	else
	{
		std::cout << "The index passed for eta_a is not valid" << std::endl;
	}
}

double N_e_a(double ksi, double eta, int a)
{
	return 0.25 * (1 + ksi_a(a) * ksi) * (1 + eta_a(a) * eta);
}

double partial_N_e_a_ksi(double ksi, double eta, int a)
{
	return 0.25 * ksi_a(a) * (1 + eta_a(a) * eta);
}

double partial_N_e_a_eta(double ksi, double eta, int a)
{
	return 0.25 * (1 + ksi_a(a) * ksi) * eta_a(a);
}

double ksi_eta_to_x(double ksi, double eta, std::array<double, 4> x_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += N_e_a(ksi, eta, i + 1) * x_i[i];
	}
	return soma;
}

double ksi_eta_to_y(double ksi, double eta, std::array<double, 4> y_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += N_e_a(ksi, eta, i + 1) * y_i[i];
	}
	return soma;
}

double partial_ksi_eta_to_x_ksi(double ksi, double eta, std::array<double, 4> x_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += partial_N_e_a_ksi(ksi, eta, i + 1) * x_i[i];
	}
	return soma;
}

double partial_ksi_eta_to_x_eta(double ksi, double eta, std::array<double, 4> x_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += partial_N_e_a_eta(ksi, eta, i + 1) * x_i[i];
	}
	return soma;
}

double partial_ksi_eta_to_y_ksi(double ksi, double eta, std::array<double, 4> y_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += partial_N_e_a_ksi(ksi, eta, i + 1) * y_i[i];
	}
	return soma;
}

double partial_ksi_eta_to_y_eta(double ksi, double eta, std::array<double, 4> y_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += partial_N_e_a_eta(ksi, eta, i + 1) * y_i[i];
	}
	return soma;
}

double partial_x_eta(double ksi, double eta, std::array<double, 4> x_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += (1 + ksi_a(i + 1) * ksi) * eta_a(i + 1) * x_i[i];
	}
	return 0.25 * soma;
}

double partial_y_eta(double ksi, double eta, std::array<double, 4> y_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += (1 + ksi_a(i + 1) * ksi) * eta_a(i + 1) * y_i[i];
	}
	return 0.25 * soma;
}

double partial_x_ksi(double ksi, double eta, std::array<double, 4> x_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += ksi_a(i + 1) * (1 + eta_a(i + 1) * eta) * x_i[i];
	}
	return 0.25 * soma;
}

double partial_y_ksi(double ksi, double eta, std::array<double, 4> y_i)
{
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += ksi_a(i + 1) * (1 + eta_a(i + 1) * eta) * y_i[i];
	}
	return 0.25 * soma;
}

Eigen::Matrix2d jacobian(double ksi, double eta, std::array<double, 4> x_i, std::array<double, 4> y_i)
{
	Eigen::Matrix2d J;
	J(0, 0) = partial_ksi_eta_to_x_ksi(ksi, eta, x_i);
	J(0, 1) = partial_ksi_eta_to_x_eta(ksi, eta, x_i);
	J(1, 0) = partial_ksi_eta_to_y_ksi(ksi, eta, y_i);
	J(1, 1) = partial_ksi_eta_to_y_eta(ksi, eta, y_i);
	return J;
}

Eigen::Matrix2d inverse_J(double ksi, double eta, std::array<double, 4> x_i, std::array<double, 4> y_i)
{
	Eigen::Matrix2d J_inv;
	J_inv(0, 0) = partial_y_eta(ksi, eta, y_i);
	J_inv(0, 1) = -1 * partial_y_ksi(ksi, eta, y_i);
	J_inv(1, 0) = -1 * partial_x_eta(ksi, eta, x_i);
	J_inv(1, 1) = partial_x_ksi(ksi, eta, x_i);
	double det = J_inv(1, 1) * J_inv(0, 0) - J_inv(1, 0) * J_inv(0, 1);
	if (det < 0.0)
	{
		std::cout << "WARNING: det < 0" << std::endl;
	}
	return (1 / det) * J_inv;
}


double gauss_point(int i)
{
	switch (i)
	{
		case 0:
			return 0.0;
		case 1:
			return GAUSS_POINT_1;
		case 2:
			return -1 * GAUSS_POINT_1;
		case 3:
			return GAUSS_POINT_2;
		case 4:
			return -1 * GAUSS_POINT_2;
		default:
			std::cout << "Invalid index for gauss_point" << std::endl;
			return 0.0;
	}
}

double gauss_weight(int i)
{
	switch (i)
	{
		case 0:
			return GAUSS_WEIGHT_0;
		case 1:
			return GAUSS_WEIGHT_1;
		case 2:
			return GAUSS_WEIGHT_1;
		case 3:
			return GAUSS_WEIGHT_2;
		case 4:
			return GAUSS_WEIGHT_2;
		default:
			std::cout << "Invalid index for gauss_weight" << std::endl;
			return 0.0;
	}
}

double interpolation_function(double x, double y, std::array<double, 4> params)
{
	return params[0] + params[1] * x + params[2] * y + params[3] * x * y;
}

double partial_interpolation_function_x(double x, double y, std::array<double, 4> params)
{
	return params[1] + params[3] * y;
}

double partial_interpolation_function_y(double x, double y, std::array<double, 4> params)
{
	return params[2] + params[3] * x;
}

Eigen::Vector2d grad_interpolation_function_xy(double x, double y, std::array<double, 4> params)
{
	Eigen::Vector2d grad;
	grad(0) = partial_interpolation_function_x(x, y, params);
	grad(1) = partial_interpolation_function_y(x, y, params);
	return grad;
}

double K_integrand(double ksi, double eta, Eigen::MatrixXd (*kappa)(double, double), std::array<double, 4> params_i, std::array<double, 4> params_j, std::array<double, 4> x_i, std::array<double, 4> y_i)
{
	double x = ksi_eta_to_x(ksi, eta, x_i);
	double y = ksi_eta_to_y(ksi, eta, y_i);
	Eigen::Vector2d grad_i = grad_interpolation_function_xy(x, y, params_i);
	Eigen::Vector2d grad_j = grad_interpolation_function_xy(x, y, params_j);
	Eigen::MatrixXd integrand = grad_i.transpose() * (kappa(x, y) * grad_j);
	Eigen::Matrix2d J =  jacobian(ksi, eta, x_i, y_i);
	double det = jacobian(ksi, eta, x_i, y_i).determinant();
	if (det <= 0)
	{
		std::cout << " WARNING:	det not positive" << std::endl;
	}
	return integrand(0) * abs(det);
}


std::array<double, 4> determine_interpolation_params(std::array<double, 4> x_i, std::array<double, 4> y_i, int index)
{
	std::array<double, 4> params;
	Eigen::Matrix<double, 4, 4> system;
	Eigen::VectorXd vec = Eigen::VectorXd::Zero(4);
	Eigen::VectorXd sol(4);
	vec(index-1) = 1.0;
	for (int i = 0; i < 4; i++)
	{
		system(i, 0) = 1.0;
		system(i, 1) = x_i[i];
		system(i, 2) = y_i[i];
		system(i, 3) = x_i[i] * y_i[i];
	}
	sol = system.colPivHouseholderQr().solve(vec);
	for (int i = 0; i < 4; i++)
	{
		params[i] = sol(i);
	}
	return params;
}

std::array<int, 4> eigen_to_int_array(Eigen::VectorXd vec)
{
	std::array<int, 4> result;
	for (int i = 0; i < 4; i++)
	{
		result[i] = vec(i);
	}
	return result;
}

class Element
{
private:
	std::array<double, 4> x_i;
	std::array<double, 4> y_i;
	std::array<std::array<double, 4>, 4> params_i;
	std::array<int, 4> local_nodes;
	std::array<int, 4> eq_indices;
	std::array<int, 4> type_of_boundary;
	Eigen::Matrix<double, 4, 4> local_k;
	Eigen::Matrix<double, 4, 4> local_m;
public:
	Eigen::Matrix<double, 4, 4> get_local_k();
	Eigen::Matrix<double, 4, 4> get_local_m();
	void set_local_matrices(double rho, double(*c_v)(double, double));
	int get_type_of_bound(int local_node);
	void set_type_of_boundary(Eigen::MatrixXd mat_of_types);
	void set_params();
	void set_coordinates(Eigen::MatrixXd mat_of_coords);
	int get_eq_index(int local_node);
	void set_eq_indices(Eigen::MatrixXd mat_of_indices);
	int get_local_node(int index);
	void set_local_nodes(Eigen::VectorXd nodes);
	std::array<double, 4> get_x_i();
	std::array<double, 4> get_y_i();
	std::array<double, 4> get_params(int index);
	Element(std::array<double, 4> xs, std::array<double, 4> ys);
	Element();
};

Eigen::Matrix<double, 4, 4> Element::get_local_k()
{
	return local_k;
}

Eigen::Matrix<double, 4, 4> Element::get_local_m()
{
	return local_m;
}


int Element::get_type_of_bound(int local_node)
{
	return type_of_boundary[local_node - 1];
}

void Element::set_type_of_boundary(Eigen::MatrixXd mat_of_types)
{
	for (int i = 0; i < 4; i++)
	{
		type_of_boundary[i] = mat_of_types(get_local_node(i + 1), 0);
	}
}

void Element::set_params()
{
	for (int i = 0; i < 4; i++)
	{
		params_i[i] = determine_interpolation_params(x_i, y_i, i + 1);
	}
}

void Element::set_coordinates(Eigen::MatrixXd mat_of_coords)
{
	for (int i = 0; i < 4; i++)
	{
		int index = get_local_node(i + 1) - 1;
		x_i[i] = mat_of_coords(index, 0);
		y_i[i] = mat_of_coords(index, 1);
	}
}

int Element::get_eq_index(int local_node)
{
	return eq_indices[local_node - 1];
}

void Element::set_eq_indices(Eigen::MatrixXd mat_of_indices)
{
	for (int i = 0; i < 4; i++)
	{
		eq_indices[i] = mat_of_indices(get_local_node(i+1), 0);
	}
}

int Element::get_local_node(int index)
{
	return local_nodes[index - 1];
}


void Element::set_local_nodes(Eigen::VectorXd nodes)
{
	local_nodes = eigen_to_int_array(nodes);
}

std::array<double, 4> Element::get_x_i()
{
	return x_i;
}

std::array<double, 4> Element::get_y_i()
{
	return y_i;
}

std::array<double, 4> Element::get_params(int index)
{
	return params_i[index];
}

Element::Element(std::array<double, 4> xs, std::array<double, 4> ys)
{
	x_i = xs;
	y_i = ys;
	for (int i = 0; i < 4; i++)
	{
		params_i[i] = determine_interpolation_params(x_i, y_i, i+1);
	}
}

Element::Element()
{
	x_i = { 0.0, 0.0, 0.0, 0.0 };
}


Eigen::Matrix<double, 4, 4> Local_K(Element local_element)
{
	Eigen::Matrix<double, 4, 4> local_k = Eigen::MatrixXd::Zero(4, 4);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			double soma = 0.0;
			for (int n = 0; n < 5; n++)
			{
				for (int m = 0; m < 5; m++)
				{
					soma += gauss_weight(n) * gauss_weight(m) * K_integrand(gauss_point(n), gauss_point(m), &KAPPA, local_element.get_params(i), local_element.get_params(j), local_element.get_x_i(), local_element.get_y_i());
				}
			}
			local_k(i, j) = soma;
		}
	}
	return local_k;
}

Eigen::SparseMatrix<double> Global_K(Element* elements,int num_el, int num_eq)
{
	Eigen::SparseMatrix<double> global_k(num_eq, num_eq);
	Eigen::Matrix<double, 4, 4> local_k;
	int row;
	int col;
	for (int e = 0; e < num_el; e++)
	{
		local_k = elements[e].get_local_k();
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				row = elements[e].get_eq_index(i + 1);
				col = elements[e].get_eq_index(j + 1);
				if (row && col)
				{
					global_k.coeffRef(row-1, col-1) += local_k(i, j);
				}
			}
		}
	}
	return global_k;
}

double M_integrand(double ksi, double eta, double rho, double(*c_v)(double, double), std::array<double, 4> params_i, std::array<double, 4> params_j, std::array<double, 4> x_i, std::array<double, 4> y_i)
{
	double x = ksi_eta_to_x(ksi, eta, x_i);
	double y = ksi_eta_to_y(ksi, eta, y_i);
	double det = jacobian(ksi, eta, x_i, y_i).determinant();
	if (det <= 0)
	{
		std::cout << " WARNING:	det not positive" << std::endl;
	}
	return rho * c_v(x, y) * interpolation_function(x, y, params_i) * interpolation_function(x, y, params_j) * abs(det);
}

Eigen::Matrix<double, 4, 4> Local_M(Element local_element, double rho, double(*c_v)(double, double))
{
	Eigen::Matrix<double, 4, 4> local_m = Eigen::MatrixXd::Zero(4, 4);
	double soma;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			soma = 0.0;
			for (int n = 0; n < 5; n++)
			{
				for (int m = 0; m < 5; m++)
				{
					soma += gauss_weight(n) * gauss_weight(m) * M_integrand(gauss_point(n), gauss_point(m), rho, c_v, local_element.get_params(i), local_element.get_params(j), local_element.get_x_i(), local_element.get_y_i());
				}
			}
			local_m(i, j) = soma;
			local_m(j, i) = soma;
		}
	}
	return local_m;
}


Eigen::SparseMatrix<double> Global_M(Element* elements, int num_el, int num_eq, double rho, double(*c_v)(double, double))
{
	Eigen::SparseMatrix<double> global_m(num_eq, num_eq);
	Eigen::Matrix<double, 4, 4> local_m;
	int row;
	int col;
	for (int e = 0; e < num_el; e++)
	{
		//local_m = Local_M(elements[e], rho, c_v);
		local_m = elements[e].get_local_m();
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				row = elements[e].get_eq_index(i + 1);
				col = elements[e].get_eq_index(j + 1);
				if (row && col)
				{
					global_m.coeffRef(row - 1, col - 1) += local_m(i, j);
				}
			}
		}
	}
	return global_m;
}

void Element::set_local_matrices(double rho, double(*c_v)(double, double))
{
	local_k = Local_K(*this);
	local_m = Local_M(*this, rho, c_v);
}

double f_first_integrand(double ksi, double eta, double t, double(*s)(double, double, double), std::array<double, 4> params, std::array<double, 4> x_i, std::array<double, 4> y_i)
{
	double x = ksi_eta_to_x(ksi, eta, x_i);
	double y = ksi_eta_to_y(ksi, eta, y_i);
	double det = jacobian(ksi, eta, x_i, y_i).determinant();
	if (det <= 0.0)
	{
		std::cout << "WARNING: det not positive" << std::endl;
	}
	return interpolation_function(x, y, params) * s(x, y, t) * abs(det);
}

double f_second_integral(Element element, int k, double t, double(*q)(double, double, double))
{
	std::array<double, 4> x_i = element.get_x_i();
	std::array<double, 4> y_i = element.get_y_i();
	double x;
	double y;
	int j;
	double w;
	double ksi;
	double eta;
	double soma = 0.0;
	double det;
	bool cond1;
	bool cond2;
	bool cond3;
	for (int i = 0; i < 4; i++)
	{
		j = (i + 1) % 4;
		{
			cond1 = element.get_type_of_bound(i + 1) == 2 && element.get_type_of_bound(j + 1) == 2;
			cond2 = element.get_type_of_bound(i + 1) == 2 && element.get_type_of_bound(j + 1) == 1;
			cond3 = element.get_type_of_bound(i + 1) == 1 && element.get_type_of_bound(j + 1) == 2;
			if (cond1 || cond2 || cond3)
			{
				for (int n = 0; n < 5; n++)
				{
					w = gauss_point(n);
					if (i == 0)
					{
						ksi = w;
						eta = -1.0;
					}
					else if (i == 1)
					{
						ksi = 1.0;
						eta = w;
					}
					else if (i == 2)
					{
						ksi = -1 * w;
						eta = 1.0;
					}
					else if (i == 3)
					{
						ksi = -1.0;
						eta = -1*w;
					}
					x = ksi_eta_to_x(ksi, eta, x_i);
					y = ksi_eta_to_y(ksi, eta, y_i);
					det = jacobian(ksi, eta, x_i, y_i).determinant();
					if (det <= 0.0)
					{
						std::cout << "WARNING: det not positive" << std::endl;
					}
					soma += gauss_weight(n) * det * interpolation_function(x, y, element.get_params(k)) * q(x, y, t);
				}
			}
		}
	}
	return soma;
}

double f_sum_term(Element element, int j, double t, double(*T)(double, double, double), double(*T_prime)(double, double, double))
{
	Eigen::Matrix<double, 4, 4> local_m = element.get_local_m();
	Eigen::Matrix<double, 4, 4> local_k = element.get_local_k();
	std::array<double, 4> x_i = element.get_x_i();
	std::array<double, 4> y_i = element.get_y_i();
	double soma = 0.0;
	for (int i = 0; i < 4; i++)
	{
		soma += local_m(j, i) * T_prime(x_i[i], y_i[i], t) - local_k(j, i) * T(x_i[i], y_i[i], t);
	}
	return soma;
}

Eigen::Vector4d Local_F(Element element, double t, double(*s)(double, double, double), double(*q)(double, double, double), double(*T)(double, double, double), double(*T_prime)(double, double, double))
{
	Eigen::Vector4d local_f = Eigen::MatrixXd::Zero(4, 1);
	double first_term;
	double second_term;
	double third_term;
	for (int i = 0; i < 4; i++)
	{
		first_term = 0.0;
		for (int n = 0; n < 5; n++)
		{
			for (int m = 0; m < 5; m++)
			{
				first_term += gauss_weight(n) * gauss_weight(m) * f_first_integrand(gauss_point(n), gauss_point(m), t, s, element.get_params(i), element.get_x_i(), element.get_y_i());
			}
		}
		second_term = f_second_integral(element, i, t, q);
		third_term = f_sum_term(element, i, t, T, T_prime);
		local_f(i) = first_term + second_term + third_term; 
	}
	return local_f;
}

Eigen::VectorXd Global_F(Element* elements, int num_el, int num_eq, double t, double(*s)(double, double, double), double(*q)(double, double, double), double(*T)(double, double, double), double(*T_prime)(double, double, double))
{
	Eigen::VectorXd global_f = Eigen::VectorXd::Zero(num_eq);
	Eigen::Vector4d local_f;
	int row;
	for (int e = 0; e < num_el; e++)
	{
		local_f = Local_F(elements[e], t, s, q, T, T_prime);
		for (int i = 0; i < 4; i++)
		{
			row = elements[e].get_eq_index(i + 1);
			if(row)
			{
				global_f(row - 1) += local_f(i);
			}
		}
	}
	return global_f;
}

Eigen::VectorXd T_zero_vec(Eigen::MatrixXd coordinates, Eigen::MatrixXd equations, int num_eqs, int num_nodes, double(*T_zero)(double, double))
{
	Eigen::VectorXd t_zero(num_eqs);
	int j = 0;
	for (int i = 0; i < num_nodes; i++)
	{
		if (equations(i+1, 0) != 0)
		{
			t_zero(j) = T_zero(coordinates(i, 0), coordinates(i, 1));
			j++;
		}
	}
	return t_zero;
}

Eigen::MatrixXd approximate_solution(double t, int num_nodes, Eigen::VectorXd coefs, Eigen::MatrixXd coords, Eigen::MatrixXd mat_of_indices, double(*T)(double, double, double))
{
	Eigen::MatrixXd approximate(num_nodes, 1);
	int row;
	for (int i = 0; i < num_nodes; i++)
	{
		row = mat_of_indices(i+1, 0);
		if(row)
		{
			approximate(i, 0) = coefs(row - 1);
		}
		else
		{
			approximate(i, 0) = T(coords(i, 0), coords(i, 1), t);
		}
	}
	return approximate;
}

Eigen::VectorXd linspace(double low, double up, int num_points)
{
	Eigen::VectorXd grid(num_points);
	double h = (up - low) / (num_points-1);
	for (int i = 0; i < num_points; i++)
	{
		grid(i) = i * h;
	}
	return grid;
}

Eigen::MatrixXd readCSV(std::string file, int rows, int cols) {

	std::ifstream in(file);

	std::string line;

	int row = 0;
	int col = 0;

	Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

	if (in.is_open()) {

		while (std::getline(in, line)) {

			char* ptr = (char*)line.c_str();
			int len = line.length();

			col = 0;

			char* start = ptr;
			for (int i = 0; i < len; i++) {

				if (ptr[i] == ',') {
					res(row, col++) = atof(start);
					start = ptr + i + 1;
				}
			}
			res(row, col) = atof(start);

			row++;
		}

		in.close();
	}
	return res;
}

void matrix_to_txt(Eigen::MatrixXd matrix, std::string name)
{
	std::ofstream file(name + ".txt");
	file.precision(40);
	int n_rows = matrix.rows();
	int n_cols = matrix.cols();
	for (int i = 0; i < n_rows; i++)
	{
		for (int j = 0; j < n_cols; j++)
		{
			file << matrix(i, j);
			if (j != n_cols - 1)
			{
				file << ", ";
			}
		}
		file << "\n";
	}
}


double T(double x, double y, double t)
{
	if (x == 0 || y == 0)
	{
		return -11.0;
	}
	if (x == 0.61 || y == 0.61)
	{
		return 20.0;
	}
	else
	{
		return 0.0;
	}
}

double T_prime(double x, double y, double t)
{
	return 0.0;
}

double s(double x, double y, double t)
{
	return 0.0;
}

double q(double x, double y, double t)
{
	return 0.5 * t;
}

double T_zero(double x, double y)
{	
	if (x == 0 || y == 0)
	{
		return -11.0;
	}
	if (x == 0.61 || y == 0.61)
	{
		return 20.0;
	}
	else
	{
		return 0.0;
	}
}

int main()
{
	std::string file_local_nodes = "elements.txt";
	std::string file_eq_matrix = "equation_indexes.txt";
	std::string file_coords = "coordinates.txt";
	std::string file_types = "type_of_boundary.txt";
	std::string file_num_nodes = "num_nodes.txt";
	std::string file_num_eqs = "num_eqs.txt";
	std::string file_num_ele = "num_elements.txt";
	int NUM_NODES = readCSV(file_num_nodes, 1, 1)(0, 0);
	int NUM_EQS = readCSV(file_num_eqs, 1, 1)(0, 0);
	int NUM_ELEMENTS = readCSV(file_num_ele, 1, 1)(0, 0);
	Eigen::MatrixXd local_nodes = readCSV(file_local_nodes, NUM_ELEMENTS, 4);
	Eigen::MatrixXd eq_matrix = readCSV(file_eq_matrix, NUM_NODES + 1, 1);
	Eigen::MatrixXd coord_matrix = readCSV(file_coords, NUM_NODES, 2);
	Eigen::MatrixXd types_matrix = readCSV(file_types, NUM_NODES + 1, 1);
	Element* elements = (Element*)malloc(NUM_ELEMENTS * sizeof(Element));
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		elements[i].set_local_nodes(local_nodes.row(i));
		elements[i].set_eq_indices(eq_matrix);
		elements[i].set_coordinates(coord_matrix);
		elements[i].set_params();
		elements[i].set_type_of_boundary(types_matrix);
		elements[i].set_local_matrices(RHO, &C_v);
		std::cout << "At element " << i << std::endl;
	}

	// The matrices
	Eigen::SparseMatrix<double> global_m = Global_M(elements, NUM_ELEMENTS, NUM_EQS, RHO, &C_v);
	Eigen::SparseMatrix<double, Eigen::ColMajor> global_k = Global_K(elements, NUM_ELEMENTS, NUM_EQS);
	global_m.makeCompressed();
	global_k.makeCompressed();
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solverK(global_k);
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solverM(global_m);
	

	// The force vectors
	Eigen::VectorXd timegrid = linspace(0.0, FINAL_TIME, NUM_STEPS);
	std::array<Eigen::MatrixXd, NUM_STEPS> approximate_sols;
	double dt = timegrid(1) - timegrid(0);
	std::array<Eigen::VectorXd, NUM_STEPS> global_fs;
	for (int i = 0; i < NUM_STEPS; i++)
	{
		global_fs[i] =  Global_F(elements, NUM_ELEMENTS, NUM_EQS, timegrid(i), &s, &q, &T, &T_prime);
		if (i % 500 == 0)
		{
			std::cout << "At step " << i << std::endl;
		}
	}

	//The stationary solution
	Eigen::VectorXd stationary = solverK.solve(global_fs[0]);
	Eigen::MatrixXd approximate_stationary = approximate_solution(0.0, NUM_NODES, stationary, coord_matrix, eq_matrix, &T);
	matrix_to_txt(approximate_stationary, "stationary_with_flux_last");
	
	// The solution in time
	Eigen::VectorXd current_T = T_zero_vec(coord_matrix, eq_matrix, NUM_EQS, NUM_NODES, &T_zero);
	approximate_sols[0] = approximate_solution(0.0, NUM_NODES, current_T, coord_matrix, eq_matrix, &T);
	
	for (int i = 1; i < NUM_STEPS; i++)
	{
		Eigen::VectorXd current_T_prime = solverM.solve(global_fs[i - 1] - global_k * current_T);
		current_T = current_T + dt * current_T_prime;
		approximate_sols[i] = approximate_solution(timegrid(i), NUM_NODES, current_T, coord_matrix, eq_matrix, &T);
		std::cout << "At step " << i << std::endl;
	}
	
	std::string file;
	for (int i = 0; i < NUM_STEPS; i++)
	{
		file = "temperature_few_steps_" + std::to_string(i);
		matrix_to_txt(approximate_sols[i], file);
		std::cout << "At step" << i << std::endl;
	}

	return 0;
}