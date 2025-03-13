#include <stdexcept>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "model.hpp"


namespace
{
    double pseudo_random(std::size_t index, std::size_t time_step) { // On donne un index et un pas de temps et on renvoie un nombre pseudo-aléatoire
        std::uint_fast32_t xi = std::uint_fast32_t(index * (time_step + 1));
        std::uint_fast32_t r = (48271 * xi) % 2147483647;
        return r / 2147483646.;
    }

    double log_factor(std::uint8_t value) {
        return std::log(1. + value) / std::log(256);
    }
}

Model::Model(double t_length, unsigned t_discretization, std::array<double, 2> t_wind, LexicoIndices t_start_fire_position, double t_max_wind)
    : m_length(t_length),
    m_distance(-1),
    m_geometry(t_discretization),
    m_wind(t_wind),
    m_wind_speed(std::sqrt(t_wind[0] * t_wind[0] + t_wind[1] * t_wind[1])),
    m_max_wind(t_max_wind),
    m_vegetation_map(t_discretization* t_discretization, 255u),
    m_fire_map(t_discretization* t_discretization, 0u) {
    if (t_discretization == 0) {
        throw std::range_error("Le nombre de cases par direction doit être plus grand que zéro.");
    }
    m_distance = m_length / double(m_geometry);
    auto index = get_index_from_lexicographic_indices(t_start_fire_position);
    m_fire_map[index] = 255u;
    m_fire_front[index] = 255u;

    constexpr double alpha0 = 4.52790762e-01;
    constexpr double alpha1 = 9.58264437e-04;
    constexpr double alpha2 = 3.61499382e-05;

    if (m_wind_speed < t_max_wind)
        p1 = alpha0 + alpha1 * m_wind_speed + alpha2 * (m_wind_speed * m_wind_speed);
    else
        p1 = alpha0 + alpha1 * t_max_wind + alpha2 * (t_max_wind * t_max_wind);
    p2 = 0.3;

    if (m_wind[0] > 0) {
        alphaEastWest = std::abs(m_wind[0] / t_max_wind) + 1;
        alphaWestEast = 1. - std::abs(m_wind[0] / t_max_wind);
    }
    else {
        alphaWestEast = std::abs(m_wind[0] / t_max_wind) + 1;
        alphaEastWest = 1. - std::abs(m_wind[0] / t_max_wind);
    }

    if (m_wind[1] > 0) {
        alphaSouthNorth = std::abs(m_wind[1] / t_max_wind) + 1;
        alphaNorthSouth = 1. - std::abs(m_wind[1] / t_max_wind);
    }
    else {
        alphaNorthSouth = std::abs(m_wind[1] / t_max_wind) + 1;
        alphaSouthNorth = 1. - std::abs(m_wind[1] / t_max_wind);
    }


    // Première étape : parallélisation
    // On initialise le vecteur des clés 


}
// --------------------------------------------------------------------------------------------------------------------
bool
Model::update() {
    // Copie de m_fire_front pour que tous les threads travaillent sur le même état initial
    std::unordered_map<std::size_t, std::uint8_t> current_front = m_fire_front;
    std::vector<std::size_t> m_keys;

    // Collection des clés pour la parallélisation
    for (const auto& f : current_front) {
        m_keys.push_back(f.first);
    }

    // On créé des containers pour stocker les chgangelments
    std::vector<std::unordered_map<std::size_t, std::uint8_t>> thread_local_additions;
    std::vector<std::vector<std::size_t>> thread_local_removals;

    #pragma omp parallel
    {
        #pragma omp single
        {
            int num_threads = omp_get_num_threads();
            thread_local_additions.resize(num_threads);
            thread_local_removals.resize(num_threads);
        }

        int thread_id = omp_get_thread_num();
        auto& local_additions = thread_local_additions[thread_id];
        auto& local_removals = thread_local_removals[thread_id];

        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < m_keys.size(); ++i) {
            std::size_t key = m_keys[i];
            std::uint8_t current_value = current_front[key];

            // Récupération de la coordonnée lexicographique de la case en feu
            LexicoIndices coord = get_lexicographic_from_index(key);
            // Et de la puissance du foyer
            double power = log_factor(current_value);

            // On va tester les cases voisines pour contamination par le feu :
            if (coord.row < m_geometry - 1) {
                double tirage = pseudo_random(key + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key + m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaSouthNorth * p1 * correction) {
                    local_additions[key + m_geometry] = 255;
                }
            }

            if (coord.row > 0) {
                double tirage = pseudo_random(key * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key - m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaNorthSouth * p1 * correction) {
                    local_additions[key - m_geometry] = 255;
                }
            }

            if (coord.column < m_geometry - 1) {
                double tirage = pseudo_random(key * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key + 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaEastWest * p1 * correction) {
                    local_additions[key + 1] = 255;
                }
            }

            if (coord.column > 0) {
                double tirage = pseudo_random(key * 13427 * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key - 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaWestEast * p1 * correction) {
                    local_additions[key - 1] = 255;
                }
            }

            // Mise à jour du feu actuel
            if (current_value == 255) {
                double tirage = pseudo_random(key * 52513 + m_time_step, m_time_step);
                if (tirage < p2) {
                    local_additions[key] = current_value >> 1;
                }
            }
            else {
                auto new_value = current_value >> 1;
                local_additions[key] = new_value;
                if (new_value == 0) {
                    local_removals.push_back(key);
                }
            }
        }
    }

    // Suppression des feux éteints


    // Mise à jour de la carte des feux
    for (auto& additions : thread_local_additions) {
        for (auto& [key, value] : additions) {
            m_fire_map[key] = value;
            m_fire_front[key] = value;
        }
    }

    for (auto& removals : thread_local_removals) {
        for (auto key : removals) {
            m_fire_map[key] = 0;
            m_fire_front.erase(key);
        }
    }

    // Mise à jour de la végétation
    for (const auto& f : m_fire_front) {
        if (m_vegetation_map[f.first] > 0)
            m_vegetation_map[f.first] -= 1;
    }

    m_time_step += 1;
    return !m_fire_front.empty();
}
// ====================================================================================================================
std::size_t
Model::get_index_from_lexicographic_indices(LexicoIndices t_lexico_indices) const {
    return t_lexico_indices.row * this->geometry() + t_lexico_indices.column;
}
// --------------------------------------------------------------------------------------------------------------------
auto
Model::get_lexicographic_from_index(std::size_t t_global_index) const -> LexicoIndices {
    LexicoIndices ind_coords;
    ind_coords.row = t_global_index / this->geometry();
    ind_coords.column = t_global_index % this->geometry();
    return ind_coords;
}
