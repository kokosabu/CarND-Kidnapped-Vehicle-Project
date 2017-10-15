/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#include <float.h>

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 128;

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    for(int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id     = i;
        particle.x      = dist_x(gen);
        particle.y      = dist_y(gen);
        particle.theta  = dist_theta(gen);
        particle.weight = 1;

        particles.push_back(particle);
        weights.push_back(particle.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
    default_random_engine gen;

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    if(fabs(yaw_rate) < 0.0001) {
        yaw_rate = 0.0001;
    }

    for(int i = 0; i < num_particles; i++) {
        double x     = particles[i].x;
        double y     = particles[i].y;
        double theta = particles[i].theta;

        double x_f     = x + velocity / yaw_rate * ( sin(theta + yaw_rate*delta_t) - sin(theta) );
        double y_f     = y + velocity / yaw_rate * ( cos(theta) - cos(theta + yaw_rate*delta_t) );
        double theta_f = theta + yaw_rate * delta_t;

        particles[i].x     = x_f     + dist_x(gen);
        particles[i].y     = y_f     + dist_y(gen);
        particles[i].theta = theta_f + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for(int i = 0; i < observations.size(); i++) {
        double dist_min = DBL_MAX;
        for(int j = 0; j < predicted.size(); j++) {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if(distance < dist_min) {
                dist_min = distance;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    for(int i = 0; i < particles.size(); i++) {
        Particle particle = particles[i];

        std::vector<LandmarkObs> mus;
        double x_part = particle.x;
        double y_part = particle.y;
        double theta  = particle.theta;
        for(int j = 0; j < observations.size(); j++) {
            double x_obs = observations[j].x;
            double y_obs = observations[j].y;

            double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs); // x_mu
            double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs); // y_mu

            LandmarkObs l;
            l.id = -1;
            l.x  = x_map;
            l.y  = y_map;

            mus.push_back(l);
        }

        std::vector<LandmarkObs> predicted;
        for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s m = map_landmarks.landmark_list[j];
            double d = dist(x_part, y_part, m.x_f, m.y_f);

            if(d < sensor_range) {
                LandmarkObs l;
                l.id = m.id_i;
                l.x  = m.x_f;
                l.y  = m.y_f;
                predicted.push_back(l);
            }
        }

        double total_weight;
        if(predicted.size() > 0) {
            dataAssociation(predicted, mus);

            total_weight = 1.0;
            for(int j = 0; j < mus.size(); j++) {
                double gauss_norm= (1.0/(2.0 * M_PI * sig_x * sig_y));
                double x_obs = mus[j].x;
                double y_obs = mus[j].y;
                double mu_x = map_landmarks.landmark_list[mus[j].id-1].x_f;
                double mu_y = map_landmarks.landmark_list[mus[j].id-1].y_f;
                double exponent = pow(x_obs - mu_x, 2)/(2 * sig_x*sig_x) + pow(y_obs - mu_y, 2)/(2 * sig_y*sig_y);

                double weight = gauss_norm * exp(-exponent);

                total_weight *= weight;
            }
        } else {
            total_weight = 0.0;
        }

        particles[i].weight = total_weight;
        weights[i]          = total_weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<Particle> new_particles;
    default_random_engine gen;

    std::discrete_distribution<> d(weights.begin(), weights.end());
    for(int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[d(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
