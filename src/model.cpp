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
    m_keys.reserve(t_discretization * t_discretization);
    for (unsigned row = 0; row < t_discretization; ++row) {
        for (unsigned column = 0; column < t_discretization; ++column) {
            auto index = get_index_from_lexicographic_indices({ row, column }); // On crée l'index
            m_keys.push_back(index); // On l'ajoute à la liste
        }
    }

}
// --------------------------------------------------------------------------------------------------------------------
bool
Model::update() {

    // Parallélisation OpenMP : on a une liste de clés et on va faire un traitement sur chaque clé


    auto next_front = m_fire_front;

    // Create thread-local containers for changes
    std::vector<std::unordered_map<std::size_t, std::uint8_t>> thread_local_additions;
    std::vector<std::vector<std::size_t>> thread_local_removals;


    // On parallélise cette boucle 
    //#pragma omp parallel 
    {

        //#pragma omp single
        {
            //printf("Running with %d threads\n", omp_get_num_threads());
            // On initialise les containers pour chaque thread
            int num_threads = omp_get_num_threads();
            thread_local_additions.resize(num_threads);
            thread_local_removals.resize(num_threads);
        }

        int thread_id = omp_get_thread_num();
        auto& local_additions = thread_local_additions[thread_id];
        auto& local_removals = thread_local_removals[thread_id];

        //#pragma omp for
        for (const auto& key : m_keys) { // On itère directement sur les clés
            //printf("Thread %d is processing key %zu\n", omp_get_thread_num(), key);

            if (m_fire_front.find(key) == m_fire_front.end())
                continue; // Si la clé n'est pas dans m_fire_front, on passe
            // Récupération de la coordonnée lexicographique de la case en feu :
            LexicoIndices coord = get_lexicographic_from_index(key);
            // Et de la puissance du foyer
            double        power = log_factor(m_fire_front[key]); // on récupère la valeur dans la carte


            // On va tester les cases voisines pour contamination par le feu :
            if (coord.row < m_geometry - 1) {
                double tirage = pseudo_random(key + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key + m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaSouthNorth * p1 * correction) {
                    m_fire_map[key + m_geometry] = 255.;
                    local_additions[key + m_geometry] = 255.;
                }
            }

            if (coord.row > 0) {
                double tirage = pseudo_random(key * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key - m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaNorthSouth * p1 * correction) {
                    m_fire_map[key - m_geometry] = 255.;
                    local_additions[key - m_geometry] = 255.;
                }
            }

            if (coord.column < m_geometry - 1) {
                double tirage = pseudo_random(key * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key + 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaEastWest * p1 * correction) {
                    m_fire_map[key + 1] = 255.;
                    local_additions[key + 1] = 255.;
                }
            }

            if (coord.column > 0) {
                double tirage = pseudo_random(key * 13427 * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[key - 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaWestEast * p1 * correction) {
                    m_fire_map[key - 1] = 255.;
                    local_additions[key - 1] = 255.;
                }
            }
            // Si le feu est à son max,
            if (m_fire_front[key] == 255) {   // On regarde si il commence à faiblir pour s'éteindre au bout d'un moment :
                double tirage = pseudo_random(key * 52513 + m_time_step, m_time_step);
                if (tirage < p2) {
                    m_fire_map[key] >>= 1;
                    local_additions[key] >>= 1;
                }
            }
            else {
                // Foyer en train de s'éteindre.
                m_fire_map[key] >>= 1;
                auto new_value = m_fire_front[key] >> 1;
                local_additions[key] = new_value;
                if (new_value == 0) {
                    local_removals.push_back(key);
                }
            }
        }
    }

    for (auto& removals : thread_local_removals) {
        for (auto key : removals) {
            next_front.erase(key);
        }
    }


    for (auto& additions : thread_local_additions) {
        for (auto& [key, value] : additions) {
            next_front[key] = value;
        }
    }


    // A chaque itération, la végétation à l'endroit d'un foyer diminue
    m_fire_front = next_front;

    for (auto key : m_keys) { // On parcourt les clés
        if (m_fire_front.find(key) == m_fire_front.end()) // Si la clé n'est pas dans m_fire_front
            continue; // On passe
        if (m_vegetation_map[key] > 0)
            m_vegetation_map[key] -= 1;
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
