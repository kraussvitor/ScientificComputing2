#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>

const double G = 6.6742e-11;
const double AU = 149597870700;
const int NUM_OF_STEPS = 2e+4;
const int FREQ = 100;
const int NUM_KNOWN_BODIES = 9;
const int NUM_ASTEROIDS = 250;
const int NUM_OF_BODIES = NUM_KNOWN_BODIES + NUM_ASTEROIDS;
const double ASTEROIDS_LOW = 450000000000.0; // 400000000000.0; //4.448e+9;
const double ASTEROIDS_UP = 600000000000.0;  // 600000000000.0; //7.48e+9;
const double ASTEROIDS_TOTAL_MASS = 2.9368e+21; // 2.9368e+21; //1 * (5.97e+24);
const double ASTEROIDS_MIN_MASS = ASTEROIDS_TOTAL_MASS / (2.0 * (NUM_ASTEROIDS+2));
const double ASTEROIDS_MAX_MASS = ASTEROIDS_TOTAL_MASS / 2.0;

typedef Eigen::Matrix<double, NUM_OF_STEPS, 1> DATA1;
typedef Eigen::Matrix<double, NUM_OF_STEPS, 3> DATA3;

std::default_random_engine generator;
std::uniform_real_distribution<double> random_dist(ASTEROIDS_LOW, ASTEROIDS_UP);
std::uniform_real_distribution<double> random_cord(-1.0, 1.0);
std::uniform_real_distribution<double> random_mass(ASTEROIDS_MIN_MASS, ASTEROIDS_MAX_MASS);


class Body
{
private:
	double mass;
	Eigen::Vector3d position;
	Eigen::Vector3d iterative_position;
	Eigen::Vector3d velocity;
	Eigen::Vector3d iterative_velocity;
	Eigen::Vector3d acceleration;

public:
	Body();
	Body(double new_mass, Eigen::Vector3d new_position, Eigen::Vector3d new_velocity);
	double get_mass();
	Eigen::Vector3d get_position();
	Eigen::Vector3d get_velocity();
	Eigen::Vector3d get_acceleration();
	Eigen::Vector3d get_iterative_position();
	Eigen::Vector3d get_iterative_velocity();
	void increment_position(Eigen::Vector3d increment);
	void increment_velocity(Eigen::Vector3d increment);
	void set_iterative_position(Eigen::Vector3d increment);
	void set_iterative_velocity(Eigen::Vector3d increment);
	void equalize_iterative_position();
	void equalize_iterative_velocity();
	void set_acceleration(Eigen::Vector3d acc);
};

Body::Body()
{
}

Body::Body(double new_mass, Eigen::Vector3d new_position, Eigen::Vector3d new_velocity)
{
	mass = new_mass;
	position = new_position;
	iterative_position = new_position;
	velocity = new_velocity;
	iterative_velocity = new_velocity;
	acceleration << 0.0, 0.0, 0.0;
}

double Body::get_mass()
{
	return mass;
}

Eigen::Vector3d Body::get_position()
{
	return position;
}

Eigen::Vector3d Body::get_velocity()
{
	return velocity;
}

Eigen::Vector3d Body::get_acceleration()
{
	return acceleration;
}

void Body::increment_position(Eigen::Vector3d increment)
{
	position += increment;
}

void Body::increment_velocity(Eigen::Vector3d increment)
{
	velocity += increment;
}

Eigen::Vector3d Body::get_iterative_position()
{
	return iterative_position;
}

Eigen::Vector3d Body::get_iterative_velocity()
{
	return iterative_velocity;
}

void Body::set_iterative_position(Eigen::Vector3d new_position)
{
	iterative_position = new_position;
}

void Body::set_iterative_velocity(Eigen::Vector3d new_velocity)
{
	iterative_velocity = new_velocity;
}

void Body::equalize_iterative_position()
{
	iterative_position = position;
}

void Body::equalize_iterative_velocity()
{
	iterative_velocity = velocity;
}

void Body::set_acceleration(Eigen::Vector3d acc)
{
	acceleration = acc;
}

// Calculates the acceleration for the index-th body in the list of bodies
Eigen::Vector3d gravitation(std::vector<Body*> bodies, int index)
{
	Eigen::Vector3d sum = Eigen::MatrixXd::Zero(3, 1);
	Eigen::Vector3d this_body_it_position = bodies[index]->get_iterative_position();
	Eigen::Vector3d difference;
	double diff_norm;
	int n = bodies.size();
	for (int i = 0; i < index; i++)
	{
		difference = bodies[i]->get_iterative_position() - this_body_it_position;
		diff_norm = difference.norm();
		sum += (bodies[i]->get_mass() / (pow(diff_norm, 3))) * difference;
	}
	for (int i = index+1; i < n; i++)
	{
		difference = bodies[i]->get_iterative_position() - this_body_it_position;
		diff_norm = difference.norm();
		sum += (bodies[i]->get_mass() / (pow(diff_norm, 3))) * difference;
	}
	return G*sum;
}

//Sums four number according to the RK4 weights
Eigen::Vector3d rk_sum(std::vector<Eigen::Vector3d> vec)
{
	//Eigen::Vector3d sum = Eigen::MatrixXd::Zero(3, 1);
	return vec[0] + 2 * vec[1] + 2 * vec[2] + vec[3];
}

//Function to simply print a Vector3d from the Eigen Lib
void print_vec(std::vector<Eigen::Vector3d> vec)
{
	int n = vec.size();
	std::cout << "vector is \n";
	for (int i = 0; i < n; i++)
	{

		std::cout << "<\n" << vec[i] << "\n>"<< std::endl;
	}
	std::cout << "endvec \n";
}

// Function that executes one single RK4 step for every body in the list of bodies
void runge_kutta_four(std::vector<Body*> bodies, double dt)
{
	int n = bodies.size();
	std::vector<std::vector<Eigen::Vector3d>> accelerations(n, std::vector<Eigen::Vector3d>(4));
	std::vector<std::vector<Eigen::Vector3d>> velocities(n, std::vector<Eigen::Vector3d>(4));
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < n; j++)
		{
			velocities[j][i] = dt*bodies[j]->get_iterative_velocity();
			accelerations[j][i] = dt*gravitation(bodies, j);
		}
		if (i < 3)
		{
			for (int j = 0; j < n; j++)
			{
				double k = 0.5;
				if (i == 2)
				{
					k = 1;
				}
				bodies[j]->set_iterative_position(bodies[j]->get_position() + k * velocities[j][i]);
				bodies[j]->set_iterative_velocity(bodies[j]->get_velocity() + k * accelerations[j][i]);
			}
		}
	}
	for (int i = 0; i < n; i++)
	{
		bodies[i]->increment_position((1.0/6.0) * rk_sum(velocities[i]));
		bodies[i]->increment_velocity((1.0/6.0) * rk_sum(accelerations[i]));
		bodies[i]->equalize_iterative_position();
		bodies[i]->equalize_iterative_velocity();
	}
}


// Function that executes one single Verlet step for each body in the list of bodies
void verlet(std::vector<Body*> bodies, double dt)
{
	int n = bodies.size();
	Eigen::Vector3d new_acceleration;
	for (int i = 0; i < n; i++)
	{
		bodies[i]->increment_position(dt * bodies[i]->get_velocity() + 0.5 * pow(dt, 2) * bodies[i]->get_acceleration());
		bodies[i]->equalize_iterative_position();
	}
	for (int i = 0; i < n; i++)
	{
		new_acceleration = gravitation(bodies, i);
		bodies[i]->increment_velocity(0.5 * dt * (bodies[i]->get_acceleration() + new_acceleration));
		bodies[i]->set_acceleration(new_acceleration);
	}
}


// Calculates the gravitational potential energy for the index-th body in the list
double potential_energy(std::vector<Body*> bodies, int index)
{
	int n = bodies.size();
	double potential_energy = 0.0;
	Eigen::Vector3d this_body_position = bodies[index]->get_position();
	double this_body_mass = bodies[index]->get_mass();
	Eigen::Vector3d diff;
	for (int i = 0; i < index; i++)
	{
		diff = this_body_position - bodies[i]->get_position();
		potential_energy -= G * this_body_mass * bodies[i]->get_mass() / diff.norm();
	}
	for (int i = index+1; i < n; i++)
	{
		diff = this_body_position - bodies[i]->get_position();
		potential_energy -= G * this_body_mass * bodies[i]->get_mass() / diff.norm();
	}
	return potential_energy;
}


// Calculates the total energy(kinetic + potential) for the index-th body in the list of bodies
double total_energy(std::vector<Body*> bodies, int index)
{
	double potential = potential_energy(bodies, index);
	double vel = 0.5 * bodies[index]->get_mass() * pow(bodies[index]->get_velocity().norm(), 2);
	return vel + potential;
}


// Calculates the angular momentum for the index-th body in the list of bodies
double angular_momentum(std::vector<Body*> bodies, int index)
{
	Eigen::Vector3d radius = bodies[index]->get_position() - bodies[0]->get_position();
	Eigen::Vector3d linear_momentum = bodies[index]->get_mass() * bodies[index]->get_velocity();
	return radius.cross(linear_momentum).norm();
}


// Calculates the velocity required to keep the index-th body in orbit
Eigen::Vector3d required_velocity(std::vector<Body*> bodies, int index)
{
	Eigen::Vector3d position = bodies[index]->get_position();
	double mass = bodies[index]->get_mass();
	Eigen::Vector3d normal;
	normal << 0.0, 0.0, 1.0;
	double potential = -1*potential_energy(bodies, index);
	double v = sqrt(potential / mass);
	Eigen::Vector3d velocity = -1*v*(position.cross(normal).normalized());
	return velocity;
}


// Create a single random position within the asteroid belt
Eigen::Vector3d random_position()
{
	double distance = random_dist(generator);
	Eigen::Vector3d position;
	position(0) = random_cord(generator);
	position(1) = random_cord(generator);
	position(2) = 0.0;
	return distance * position.normalized();
}


// Create many random positions within the asteroid belt
std::vector<Eigen::Vector3d> random_positions(int num)
{
	std::vector<Eigen::Vector3d> positions(num);
	for (int i = 0; i < num; i++)
	{
		positions[i] = random_position();
	}
	return positions;
}


// Create a vector of masses that sum up to the given total
std::vector<double> random_masses(int num, double total)
{
	std::vector<double> masses(num); 
	double total_mass = 0.0;
	double mass;
	for (int i = 0; i < num; i++)
	{
		mass = random_mass(generator);
		masses[i] = mass;
		total_mass += mass;
	}
	for (int i = 0; i < num; i++)
	{
		masses[i] = masses[i] / total_mass;
	}
	return masses;
}


// Gets a Matrix(from Eigen Lib) and writes it into a .txt file 
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

// Executes the simulation for a number of steps and storages the positions, energies and momentums of each body at each step
void simulate(std::vector<Body*> bodies, int num_of_steps, double dt, void (*f)(std::vector<Body*>, double), std::string method_name)
{
	int n = bodies.size();
	DATA1* kinetic_data = (DATA1*)malloc(NUM_KNOWN_BODIES * sizeof(DATA1));
	DATA1* angular_data = (DATA1*)malloc(NUM_KNOWN_BODIES * sizeof(DATA1));
	DATA3* data = (DATA3*)malloc(n*sizeof(DATA3));
	std::string filename;
	for (int i = 0; i < num_of_steps; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Eigen::Vector3d pos = bodies[j]->get_position();
			data[j](i, 0) = pos(0);
			data[j](i, 1) = pos(1);
			data[j](i, 2) = pos(2);
			if (j < NUM_KNOWN_BODIES)
			{
				kinetic_data[j](i, 0) = total_energy(bodies, j);
				angular_data[j](i, 0) = angular_momentum(bodies, j);
			}
		}

		(*f)(bodies, dt);
		
		if (i % FREQ == 0)
		{
			std::cout << "at step " << i << std::endl;
		}
	}
	for (int i = 0; i < n; i++)
	{
		filename = method_name + "_body_" + std::to_string(i);
		matrix_to_txt(data[i], filename);
	}
	for (int i = 0; i < NUM_KNOWN_BODIES; i++)
	{
		filename = method_name + "_total_energy_" + std::to_string(i);
		matrix_to_txt(kinetic_data[i], filename);
		filename = method_name + "_angular_momentum_" + std::to_string(i);
		matrix_to_txt(angular_data[i], filename);
	}
	free(kinetic_data);
	free(angular_data);
	free(data);
}

// Calls the simulation function with the given method and also measure its execution time
void call_simulation(std::vector<Body*> bodies, int num_of_steps, double dt, std::string method_name, std::string prefix)
{
	auto start = std::chrono::system_clock::now();
	if (method_name == "runge_kutta")
	{
		simulate(bodies, num_of_steps, dt, &runge_kutta_four, prefix + method_name);
	}
	else if (method_name == "verlet")
	{
		simulate(bodies, num_of_steps, dt, &verlet, prefix + method_name);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::string time = prefix + method_name + "_time";
	std::ofstream file(time + ".txt");
	file << "Method: " << method_name << " \n";
	file << "Baking time = " << elapsed_seconds.count() << " seconds \n";
	file << "with timestep = " << dt << " seconds \n";
	file << "for " << NUM_OF_STEPS << " steps \n";
	file << "with " << NUM_OF_BODIES << " bodies \n";
}


// Convert astronomical units to meters
double au_to_meters(double x)
{
	return AU * x;
}

// Convert a vector in au to meters
Eigen::Vector3d au_to_meters(Eigen::Vector3d vec)
{
	Eigen::Vector3d new_vec;
	new_vec(0) = au_to_meters(vec(0));
	new_vec(1) = au_to_meters(vec(1));
	new_vec(2) = au_to_meters(vec(2));
	return new_vec;
}

//Convert astronomical units per day to meters per second
double aud_to_ms(double x)
{
	return 1.731e+6 * x;
}

//Convert a vector in au/d to m/s
Eigen::Vector3d aud_to_ms(Eigen::Vector3d vec)
{
	Eigen::Vector3d new_vec;
	new_vec(0) = aud_to_ms(vec(0));
	new_vec(1) = aud_to_ms(vec(1));
	new_vec(2) = aud_to_ms(vec(2));
	return new_vec;
}

int main()
{
	std::vector<Body*> bodies(NUM_OF_BODIES);
	std::vector<double> known_masses{ 1.9885 * pow(10, 30) , 3.3011 * pow(10, 23), 4.8674 * pow(10, 24), 5.97237 * pow(10, 24), 6.4171 * pow(10, 23), 1.8982 * pow(10, 27), 5.6834 * pow(10, 26), 8.6810 * pow(10, 25), 1.02413 * pow(10, 26) };
	std::vector<Eigen::Vector3d> known_positions(NUM_KNOWN_BODIES);
	std::vector<Eigen::Vector3d> known_velocities(NUM_KNOWN_BODIES);
	Eigen::Vector3d zero_vec;
	zero_vec << 0.0, 0.0, 0.0;
	known_positions[0] << 0.0, 0.0, 0.0;
	known_positions[1] << -1.26918062409295e-1, -4.47750819828178e-1, -3.17281414202121e-2;
	known_positions[2] << -7.18938244593697e-1, -3.68306028009161e-2, 2.18621410503987e-2;
	known_positions[3] << -1.82442732168004e-1, 9.66248614272794e-1, 3.39363083867148e-3;
	known_positions[4] << 1.39119459867718, -5.70724722818469e-3, 1.95802300603516e-3;
	known_positions[5] << +3.98608854049054, 2.96063149041822, 2.77925043901118e-2;
	known_positions[6] << 6.37729717199565, 6.60695460327579, -1.45998773495659e-1;
	known_positions[7] << 1.45092025008910e+1, -1.36546370106073e+1, 2.66757061311420e-2;
	known_positions[8] << 1.69414960422677e+1, -2.49019458134467e+1, 3.60724496679461e-1;
	known_velocities[0] << 0.0, 0.0, 0.0;
	known_velocities[1] << 2.14597548234309e-2, -6.31265972438156e-3, -1.97943813468118e-3;
	known_velocities[2] << 9.19016637306985e-4, -2.02872083037635e-2, -4.71273188233886e-4;
	known_velocities[3] << -1.71839053754536e-2, -3.25039279379074e-3, -4.78106199576676e-4;
	known_velocities[4] << 5.79011526780407e-4, 1.51875155841343e-2, 4.45567579943964e-4;
	known_velocities[5] << -4.60443799817529e-3, 6.41787768831458e-3, 9.19511388697671e-6;
	known_velocities[6] << -4.31511379100557e-3, 3.86635585175966e-3, 2.25549068246028e-5;
	known_velocities[7] << 2.66294837758073e-3, 2.68705672751516e-3, 6.78162029611345e-5;
	known_velocities[8] << 2.57099524793415e-3, 1.79137641480837e-3, -1.33438959942056e-5;
	for (int i = 0; i < NUM_KNOWN_BODIES; i++)
	{
		known_positions[i] = au_to_meters(known_positions[i]);
		known_velocities[i] = aud_to_ms(known_velocities[i]);
		bodies[i] = new Body(known_masses[i], known_positions[i], known_velocities[i]);
	}
	if (NUM_ASTEROIDS > 0)
	{
		std::vector<Eigen::Vector3d> rand_positions = random_positions(NUM_ASTEROIDS);
		std::vector<double> rand_masses = random_masses(NUM_ASTEROIDS, ASTEROIDS_TOTAL_MASS);
		int j;
		for (int i = 0; i < NUM_ASTEROIDS; i++)
		{
			j = i + NUM_KNOWN_BODIES;
			bodies[j] = new Body(rand_masses[i], rand_positions[i], zero_vec);
		}
		Eigen::Vector3d veloc;
		for (int i = NUM_KNOWN_BODIES; i < NUM_OF_BODIES; i++)
		{
			veloc = required_velocity(bodies, i);
			bodies[i]->increment_velocity(veloc);
		}
	}
	std::string method = "verlet";
	double h = 1e+4;
	std::string prefix = "with_asteroids_" + std::to_string(NUM_ASTEROIDS) + "_";
	call_simulation(bodies, NUM_OF_STEPS, h, method, prefix);
	return 0;
}