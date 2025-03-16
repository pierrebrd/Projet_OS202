#include <stdexcept>
#include <cmath>
#include <iostream>
#include <mpi.h>
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
}

int mod(int a, int b) {
    return (a % b + b) % b; // Ensures a positive remainder
}

void 
Model::update_ghost_cells(int rank, int n_rank, MPI_Comm newCom){
    // On procède à un découpage en tranche selon l'axe des ordonnées en répartissant le reste r entre les premières tranches
    // On commence par calculer la taille de chaque tranche
    int r = m_geometry % n_rank;
    int size = m_geometry / n_rank;
    int start_y = rank * size + std::min(rank, r); // Coord y de départ
    int end_y = start_y + size + (rank < r); // Coord y de fin
    int start = start_y * m_geometry;
    int end = end_y * m_geometry;

    // Partie pour le feu
    std::vector<uint8_t> ghost_cells_up(m_geometry, 0u);
    std::vector<uint8_t> ghost_cells_down(m_geometry, 0u);

    std::vector<uint8_t> sending_cells_up(m_fire_map.begin() + start, m_fire_map.begin() + start + m_geometry);
    std::vector<uint8_t> sending_cells_down(m_fire_map.begin() + end - m_geometry, m_fire_map.begin() + end);

    MPI_Request req1;
    MPI_Request req2;
    MPI_Irecv(ghost_cells_up.data(), m_geometry, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom, &req1);                
    MPI_Irecv(ghost_cells_down.data(), m_geometry, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom, &req2);                
    
    MPI_Send(sending_cells_up.data(), m_geometry, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom);
    MPI_Send(sending_cells_down.data(), m_geometry, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom);

    MPI_Wait(&req1, MPI_STATUS_IGNORE);
    MPI_Wait(&req2, MPI_STATUS_IGNORE);

    if(rank != 0){
        std::copy(ghost_cells_up.begin(), ghost_cells_up.end(), m_fire_map.begin() + start - m_geometry);
    }
    if(rank != n_rank - 1){
        std::copy(ghost_cells_down.begin(), ghost_cells_down.end(), m_fire_map.begin() + end);
    }

    // Partie pour le végetal
    std::vector<uint8_t> ghost_cells_up_vegetal(m_geometry, 0u);
    std::vector<uint8_t> ghost_cells_down_vegetal(m_geometry, 0u);

    std::vector<uint8_t> sending_cells_up_vegetal(m_vegetation_map.begin() + start, m_vegetation_map.begin() + start + m_geometry);
    std::vector<uint8_t> sending_cells_down_vegetal(m_vegetation_map.begin()+ end - m_geometry, m_vegetation_map.begin() + end);

    MPI_Request req3;
    MPI_Request req4;
    MPI_Irecv(ghost_cells_up_vegetal.data(), m_geometry, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom, &req3);
    MPI_Irecv(ghost_cells_down_vegetal.data(), m_geometry, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom, &req4);
    
    MPI_Send(sending_cells_up_vegetal.data(), m_geometry, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom);
    MPI_Send(sending_cells_down_vegetal.data(), m_geometry, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom);

    MPI_Wait(&req3, MPI_STATUS_IGNORE);
    MPI_Wait(&req4, MPI_STATUS_IGNORE);

    if(rank != 0){
        std::copy(ghost_cells_up_vegetal.begin(), ghost_cells_up_vegetal.end(), m_vegetation_map.begin() + start - m_geometry);
    }
    if(rank != n_rank - 1){
        std::copy(ghost_cells_down_vegetal.begin(), ghost_cells_down_vegetal.end(), m_vegetation_map.begin() + end);
    }

    // Partie pour le fire_front
    // Collection des clés pour la parallélisation : on récupère et envoie les coordonnées des cases en feu adjacentes
    MPI_Request req5;
    MPI_Request req6;
    
    std::vector<std::size_t> sending_keys_up;
    std::vector<std::size_t> sending_keys_down;
    std::vector<uint8_t> sending_values_up;
    std::vector<uint8_t> sending_values_down;

    std::unordered_map<std::size_t, std::uint8_t> current_front = m_fire_front;
    for (const auto& f : current_front) {
        if((f.first >= start) && (f.first < start + m_geometry)){
            sending_keys_up.push_back(f.first);
            sending_values_up.push_back(f.second);
        }
        if((f.first >= end - m_geometry) && (f.first < end)){
            sending_keys_down.push_back(f.first);
            sending_values_down.push_back(f.second);
        }
    }

    // On envoie et reçoit les tailles des vecteurs
    int size_up = sending_keys_up.size();
    int size_down = sending_keys_down.size();

    int size_ghost_up = 0;
    int size_ghost_down = 0;
    MPI_Irecv(&size_ghost_up, 1, MPI_INT, mod(rank-1, n_rank), 0, newCom, &req5);
    MPI_Irecv(&size_ghost_down, 1, MPI_INT, (rank+1)%n_rank, 0, newCom, &req6);

    MPI_Send(&size_up, 1, MPI_INT, mod(rank-1, n_rank), 0, newCom);
    MPI_Send(&size_down, 1, MPI_INT, (rank+1)%n_rank, 0, newCom);

    MPI_Wait(&req5, MPI_STATUS_IGNORE);
    MPI_Wait(&req6, MPI_STATUS_IGNORE);

    // On envoie et reçoit les clés
    std::vector<std::size_t> ghost_fire_up(size_ghost_up, 0u);
    std::vector<std::size_t> ghost_fire_down(size_ghost_down, 0u);

    MPI_Request req7;
    MPI_Request req8;

    MPI_Irecv(ghost_fire_up.data(), size_ghost_up, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom, &req7);
    MPI_Irecv(ghost_fire_down.data(), size_ghost_down, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom, &req8);

    MPI_Send(sending_keys_up.data(), size_up, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom);
    MPI_Send(sending_keys_down.data(), size_down, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom);

    MPI_Wait(&req7, MPI_STATUS_IGNORE);
    MPI_Wait(&req8, MPI_STATUS_IGNORE);

    // On envoie et reçoit les valeurs associées aux clés
    std::vector<uint8_t> ghost_fire_up_values(size_ghost_up, 0u);
    std::vector<uint8_t> ghost_fire_down_values(size_ghost_down, 0u);

    MPI_Request req9;
    MPI_Request req10;

    MPI_Irecv(ghost_fire_up_values.data(), size_ghost_up, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom, &req9);
    MPI_Irecv(ghost_fire_down_values.data(), size_ghost_down, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom, &req10);

    MPI_Send(sending_values_up.data(), size_up, MPI_UINT8_T, mod(rank-1, n_rank), 0, newCom);
    MPI_Send(sending_values_down.data(), size_down, MPI_UINT8_T, (rank+1)%n_rank, 0, newCom);

    MPI_Wait(&req9, MPI_STATUS_IGNORE);
    MPI_Wait(&req10, MPI_STATUS_IGNORE);

    // On met à jour les ghost_cells
    if(rank != 0){
        for(int i = 0; i < size_ghost_up; i++){
            m_fire_front[ghost_fire_up[i]] = ghost_fire_up_values[i];
        }
    }
    if(rank != n_rank - 1){
        for(int i = 0; i < size_ghost_down; i++){
            m_fire_front[ghost_fire_down[i]] = ghost_fire_down_values[i];
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------
bool
Model::update() {
    auto next_front = m_fire_front;

    for (auto f : m_fire_front) {
        // Récupération de la coordonnée lexicographique de la case en feu :
        LexicoIndices coord = get_lexicographic_from_index(f.first);
        // Et de la puissance du foyer
        double        power = log_factor(f.second);


        // On va tester les cases voisines pour contamination par le feu :
        if (coord.row < m_geometry - 1) {
            double tirage = pseudo_random(f.first + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaSouthNorth * p1 * correction) {
                m_fire_map[f.first + m_geometry] = 255.;
                next_front[f.first + m_geometry] = 255.;
            }
        }

        if (coord.row > 0) {
            double tirage = pseudo_random(f.first * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaNorthSouth * p1 * correction) {
                m_fire_map[f.first - m_geometry] = 255.;
                next_front[f.first - m_geometry] = 255.;
            }
        }

        if (coord.column < m_geometry - 1) {
            double tirage = pseudo_random(f.first * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaEastWest * p1 * correction) {
                m_fire_map[f.first + 1] = 255.;
                next_front[f.first + 1] = 255.;
            }
        }

        if (coord.column > 0) {
            double tirage = pseudo_random(f.first * 13427 * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaWestEast * p1 * correction) {
                m_fire_map[f.first - 1] = 255.;
                next_front[f.first - 1] = 255.;
            }
        }
        // Si le feu est à son max,
        if (f.second == 255) {   // On regarde si il commence à faiblir pour s'éteindre au bout d'un moment :
            double tirage = pseudo_random(f.first * 52513 + m_time_step, m_time_step);
            if (tirage < p2) {
                m_fire_map[f.first] >>= 1;
                next_front[f.first] >>= 1;
            }
        }
        else {
            // Foyer en train de s'éteindre.
            m_fire_map[f.first] >>= 1;
            next_front[f.first] >>= 1;
            if (next_front[f.first] == 0) {
                next_front.erase(f.first);
            }
        }

    }
    // A chaque itération, la végétation à l'endroit d'un foyer diminue
    m_fire_front = next_front;
    for (auto f : m_fire_front) {
        if (m_vegetation_map[f.first] > 0)
            m_vegetation_map[f.first] -= 1;
    }
    m_time_step += 1;
    return !m_fire_front.empty();
}

bool
Model::update(int rank, int n_rank, MPI_Comm newCom) {
    // On procède à un découpage en tranche selon l'axe des ordonnées en répartissant le reste r entre les premières tranches
    // On commence par calculer la taille de chaque tranche
    int r = m_geometry % n_rank;
    int size = m_geometry / n_rank;
    int start_y = rank * size + std::min(rank, r);
    int end_y = start_y + size + (rank < r);
    int start = start_y * m_geometry;
    int end = end_y * m_geometry;

    // On récupère les ghost_cells
    update_ghost_cells(rank, n_rank, newCom);

    // On met à jour les cases de la tranche qui nous intéresse
    auto next_front = m_fire_front;
    for (auto f : m_fire_front) {
        // auto f = m_fire_front[i];
        // Récupération de la coordonnée lexicographique de la case en feu :
        LexicoIndices coord = get_lexicographic_from_index(f.first);
        if((f.first >= start)&&(f.first < end)){
            // Et de la puissance du foyer
            double        power = log_factor(f.second);
    
            // On va tester les cases voisines pour contamination par le feu :
            if (coord.row < m_geometry - 1) {
                double tirage = pseudo_random(f.first + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first + m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaSouthNorth * p1 * correction) {
                    m_fire_map[f.first + m_geometry] = 255.;
                    next_front[f.first + m_geometry] = 255.;
                }
            }
    
            if (coord.row > 0) {
                double tirage = pseudo_random(f.first * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first - m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaNorthSouth * p1 * correction) {
                    m_fire_map[f.first - m_geometry] = 255.;
                    next_front[f.first - m_geometry] = 255.;
                }
            }
    
            if (coord.column < m_geometry - 1) {
                double tirage = pseudo_random(f.first * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first + 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaEastWest * p1 * correction) {
                    m_fire_map[f.first + 1] = 255.;
                    next_front[f.first + 1] = 255.;
                }
            }
    
            if (coord.column > 0) {
                double tirage = pseudo_random(f.first * 13427 * 13427 * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first - 1];
                double correction = power * log_factor(green_power);
                if (tirage < alphaWestEast * p1 * correction) {
                    m_fire_map[f.first - 1] = 255.;
                    next_front[f.first - 1] = 255.;
                }
            }
            // Si le feu est à son max,
            if (f.second == 255) {   // On regarde si il commence à faiblir pour s'éteindre au bout d'un moment :
                double tirage = pseudo_random(f.first * 52513 + m_time_step, m_time_step);
                if (tirage < p2) {
                    m_fire_map[f.first] >>= 1;
                    next_front[f.first] >>= 1;
                }
            }
            else {
                // Foyer en train de s'éteindre.
                m_fire_map[f.first] >>= 1;
                next_front[f.first] >>= 1;
                if (next_front[f.first] == 0) {
                    next_front.erase(f.first);
                }
            }
        }
        if ((coord.row == (start_y-1))&&(rank != 0)){
            // On se trouve dans une case enflammé sur la ligne au dessus de la tranche
            double power = log_factor(f.second);
            if (coord.row < m_geometry - 1) { // Si on ne se trouve pas sur la dernière ligne
                double tirage = pseudo_random(f.first + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first + m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaSouthNorth * p1 * correction) {
                    m_fire_map[f.first + m_geometry] = 255.;
                    next_front[f.first + m_geometry] = 255.;
                }
            }
        }
        if ((coord.row == end_y)&&(rank != n_rank-1)){
            // On se trouve dans une case enflammé sur la ligne au dessus de la tranche
            double power = log_factor(f.second);
            if (coord.row > 0) {
                double tirage = pseudo_random(f.first * 13427 + m_time_step, m_time_step);
                double green_power = m_vegetation_map[f.first - m_geometry];
                double correction = power * log_factor(green_power);
                if (tirage < alphaNorthSouth * p1 * correction) {
                    m_fire_map[f.first - m_geometry] = 255.;
                    next_front[f.first - m_geometry] = 255.;
                }
            }
        }
    }
    // A chaque itération, la végétation à l'endroit d'un foyer diminue
    m_fire_front = next_front;
    for (auto it = m_fire_front.begin(); it != m_fire_front.end(); ) {
        if (it->first >= start && it->first < end) {
            if (m_vegetation_map[it->first] > 0)
                m_vegetation_map[it->first] -= 1;
            ++it;
        } else {
            it = m_fire_front.erase(it);  // erase retourne l'itérateur vers l'élément suivant
        }
    }
    m_time_step += 1;
    return !m_fire_front.empty(); // Problème ici, on risque de ne pas avoir un front empty si on laisse des cases bonus
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
