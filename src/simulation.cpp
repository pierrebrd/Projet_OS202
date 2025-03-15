#include <string>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>
#include <mpi.h>


#include "model.hpp"
#include "display.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

struct ParamsType // Paramètres de la simulation
{
    double length{ 1. };
    unsigned discretization{ 20u };
    std::array<double, 2> wind{ 0.,0. };
    Model::LexicoIndices start{ 10u,10u };
};

void analyze_arg(int nargs, char* args[], ParamsType& params) {
    if (nargs == 0) return;
    std::string key(args[0]);
    if (key == "-l"s) {
        if (nargs < 2) {
            std::cerr << "Manque une valeur pour la longueur du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.length = std::stoul(args[1]);
        analyze_arg(nargs - 2, &args[2], params);
        return;
    }
    auto pos = key.find("--longueur=");
    if (pos < key.size()) {
        auto subkey = std::string(key, pos + 11);
        params.length = std::stoul(subkey);
        analyze_arg(nargs - 1, &args[1], params);
        return;
    }

    if (key == "-n"s) {
        if (nargs < 2) {
            std::cerr << "Manque une valeur pour le nombre de cases par direction pour la discrétisation du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.discretization = std::stoul(args[1]);
        analyze_arg(nargs - 2, &args[2], params);
        return;
    }
    pos = key.find("--number_of_cases=");
    if (pos < key.size()) {
        auto subkey = std::string(key, pos + 18);
        params.discretization = std::stoul(subkey);
        analyze_arg(nargs - 1, &args[1], params);
        return;
    }

    if (key == "-w"s) {
        if (nargs < 2) {
            std::cerr << "Manque une paire de valeurs pour la direction du vent !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values = std::string(args[1]);
        params.wind[0] = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size()) {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos + 1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs - 2, &args[2], params);
        return;
    }
    pos = key.find("--wind=");
    if (pos < key.size()) {
        auto subkey = std::string(key, pos + 7);
        params.wind[0] = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size()) {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos + 1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs - 1, &args[1], params);
        return;
    }

    if (key == "-s"s) {
        if (nargs < 2) {
            std::cerr << "Manque une paire de valeurs pour la position du foyer initial !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values = std::string(args[1]);
        params.start.column = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size()) {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la position du foyer initial" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos + 1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs - 2, &args[2], params);
        return;
    }
    pos = key.find("--start=");
    if (pos < key.size()) {
        auto subkey = std::string(key, pos + 8);
        params.start.column = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size()) {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos + 1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs - 1, &args[1], params);
        return;
    }
}

ParamsType parse_arguments(int nargs, char* args[]) {
    if (nargs == 0) return {};
    if ((std::string(args[0]) == "--help"s) || (std::string(args[0]) == "-h")) {
        std::cout <<
            R"RAW(Usage : simulation [option(s)]
  Lance la simulation d'incendie en prenant en compte les [option(s)].
  Les options sont :
    -l, --longueur=LONGUEUR     Définit la taille LONGUEUR (réel en km) du carré représentant la carte de la végétation.
    -n, --number_of_cases=N     Nombre n de cases par direction pour la discrétisation
    -w, --wind=VX,VY            Définit le vecteur vitesse du vent (pas de vent par défaut).
    -s, --start=COL,ROW         Définit les indices I,J de la case où commence l'incendie (milieu de la carte par défaut)
)RAW";
        exit(EXIT_SUCCESS);
    }
    ParamsType params;
    analyze_arg(nargs, args, params);
    return params;
}

bool check_params(ParamsType& params) {
    bool flag = true;
    if (params.length <= 0) {
        std::cerr << "[ERREUR FATALE] La longueur du terrain doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if (params.discretization <= 0) {
        std::cerr << "[ERREUR FATALE] Le nombre de cellules par direction doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if ((params.start.row >= params.discretization) || (params.start.column >= params.discretization)) {
        std::cerr << "[ERREUR FATALE] Mauvais indices pour la position initiale du foyer" << std::endl;
        flag = false;
    }

    return flag;
}

void display_params(ParamsType const& params) {
    std::cout << "Parametres définis pour la simulation : \n"
        << "\tTaille du terrain : " << params.length << std::endl
        << "\tNombre de cellules par direction : " << params.discretization << std::endl
        << "\tVecteur vitesse : [" << params.wind[0] << ", " << params.wind[1] << "]" << std::endl
        << "\tPosition initiale du foyer (col, ligne) : " << params.start.column << ", " << params.start.row << std::endl;
}

// Fonction générique pour mesurer le temps d'exécution d'une méthode
template<typename Obj, typename Method, typename... Args>
auto measure_time(bool condition, Obj&& objet, Method&& methode, Args&&... args) {
    if(condition){
        auto start = std::chrono::high_resolution_clock::now();
    
        auto result = (std::forward<Obj>(objet).*std::forward<Method>(methode))(std::forward<Args>(args)...);
    
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
        std::cout << "Temps d'exécution : " << duration.count() << " microsecondes" << std::endl;
        return result;
    }
    else{
        auto result = (std::forward<Obj>(objet).*std::forward<Method>(methode))(std::forward<Args>(args)...);
        return result;
    }
}

int main(int nargs, char* args[]) {
    // Initialisation de MPI
    MPI_Init(&nargs, &args);

    // Obtenir le rang du processus courant
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Obtenir le nombre total de processus
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Process " << rank << " out of " << size << " is running." << std::endl;
    
    // Sous-communicateur pour le calcul parallèle de la simulation.
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank != 0, rank, &new_comm);

    // Obtenir le nouveau rang dans le sous-communicateur
    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);
    std::cout << "Process " << new_rank << " out of " << new_size << " is running." << std::endl;

    auto params = parse_arguments(nargs - 1, &args[1]);
    display_params(params);
    if (!check_params(params)) return EXIT_FAILURE;

    int discretization = params.discretization;
    // auto simu = Model(params.length, params.discretization, params.wind, params.start); // On lance la simulation
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if(size >= 2){
        int table_size = params.discretization * params.discretization;
        if(rank == 0){
            // Thread s'occupant de l'affichage
            auto displayer = Displayer::init_instance(params.discretization, params.discretization); // On lance la fenêtre d'affichage
            SDL_Event event;

            std::vector<uint8_t> vegetal_map(table_size, 255u);
            std::vector<uint8_t> fire_map(table_size, 0u);
            MPI_Status status;

            MPI_Recv(vegetal_map.data(), vegetal_map.size() , MPI_UINT8_T, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(fire_map.data(), fire_map.size(), MPI_UINT8_T, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            while(status.MPI_TAG != 1){
                displayer->update(vegetal_map, fire_map);
                MPI_Recv(vegetal_map.data(), vegetal_map.size() , MPI_UINT8_T, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                MPI_Recv(fire_map.data(), fire_map.size(), MPI_UINT8_T, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
                    break;
                std::this_thread::sleep_for(0.1s);
            }
        }
        else{
            MPI_Request request1;
            MPI_Request request2;

            // Params du thread
            int n_rank = new_size;
            int r = params.discretization % n_rank;
            int size = params.discretization / n_rank;
            int start_y = new_rank * new_size + std::min(new_rank, r);
            int end_y = start_y + size + (rank < r);
            int start = start_y * params.discretization;
            int end = end_y * params.discretization;

            // Créer la grille associée au processus. La grille est de taille maximale mais seule la partie associée au processus est modifiée.
            // Attention à ce que toutes les initialisations soient strictement identiques pour tous les processus !
            // Faire attention au pseudo aléatoires dans les différents processus.
            auto simu = Model(params.length, params.discretization, params.wind, params.start); // On lance la simulation

            while(1){
                // Envoyer les résultats au processus 0' pour gather l'ensemble dans une seule grille
                int total_size = simu.fire_map().size();  // Taille totale du vecteur

                if (start < 0 || end > total_size || start >= end) {
                    std::cerr << "[ERREUR] Indices invalides : start=" << start << ", end=" << end << ", taille=" << total_size << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    return EXIT_FAILURE;
                }
                
                std::vector<uint8_t> small_fire_map(simu.fire_map().begin() + start, simu.fire_map().begin() + end);
                std::vector<uint8_t> small_vegetal_map(simu.vegetal_map().begin() + start, simu.vegetal_map().begin() + end);
    
                // Définition des tailles d'envoi pour chaque processus
                int local_fire_size = end - start;
                int local_vegetal_size = end - start;

                // Collecter les tailles d'envoi sur le processus root
                std::vector<int> recv_counts(size, 0);
                MPI_Gather(&local_fire_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, new_comm);

                // Calculer les offsets (`displs`) sur le processus root
                std::vector<int> displs(size, 0);
                if (new_rank == 0) {
                    for (int i = 1; i < size; i++) {
                        displs[i] = displs[i - 1] + recv_counts[i - 1];
                    }
                }

                // Allocation des buffers finaux sur le processus root
                std::vector<uint8_t> fire_map;
                std::vector<uint8_t> vegetal_map;
                if (new_rank == 0) {
                    int total_size = displs[size - 1] + recv_counts[size - 1];  // Taille totale requise
                    fire_map.resize(total_size);
                    vegetal_map.resize(total_size);
                }

                // Exécuter `MPI_Gatherv`
                MPI_Gatherv(small_fire_map.data(), local_fire_size, MPI_UINT8_T,
                            fire_map.data(), recv_counts.data(), displs.data(), MPI_UINT8_T,
                            0, new_comm);

                MPI_Gatherv(small_vegetal_map.data(), local_vegetal_size, MPI_UINT8_T,
                            vegetal_map.data(), recv_counts.data(), displs.data(), MPI_UINT8_T,
                            0, new_comm);
    
                // Mettre à jour le modèle de 0' et l'envoyer au 0 pour affichage
                if (new_rank == 0){
                    MPI_Isend(vegetal_map.data(), vegetal_map.size(), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &request1);                
                    MPI_Wait(&request1, MPI_STATUS_IGNORE);
                    MPI_Isend(fire_map.data(), fire_map.size(), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &request2);
                    MPI_Wait(&request2, MPI_STATUS_IGNORE);
                }
                // Faire le calcul
                bool run = simu.update(new_rank, new_size, new_comm);
            }
                // MPI_Isend(vegetal_map.data(), vegetal_map.size(), MPI_UINT8_T, 0, 1, MPI_COMM_WORLD, &request);
                // MPI_Wait(&request, MPI_STATUS_IGNORE);
                // MPI_Isend(fire_map.data(), fire_map.size(), MPI_UINT8_T, 0, 1, MPI_COMM_WORLD, &request);
                // MPI_Wait(&request, MPI_STATUS_IGNORE);
            // while (measure_time(((simu.time_step() & 31) == 0), simu, &Model::update)){
            // }
        }
    }
    else {
        auto displayer = Displayer::init_instance( params.discretization, params.discretization );
        auto simu = Model(params.length, params.discretization, params.wind, params.start); // On lance la simulation
        SDL_Event event;
        while (measure_time(((simu.time_step() & 31) == 0), simu, static_cast<bool (Model::*)()>(&Model::update))){ // Modification de la fonction pour mesurer le temps d'exécution
            if ((simu.time_step() & 31) == 0){
                std::cout << "Time step " << simu.time_step() << "\n===============" << std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                displayer->update(simu.vegetal_map(), simu.fire_map()); // On met à jour l'affichage
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            
                std::cout << "Temps d'exécution affichage: " << duration.count() << " microsecondes" << std::endl;
            }
            else{
                displayer->update(simu.vegetal_map(), simu.fire_map()); // On met à jour l'affichage
            }
            //measure_time(((simu.time_step() & 31) == 0), displayer, &Displayer::update, simu.vegetal_map(), simu.fire_map()); // Modification de la fonction pour mesurer le temps d'exécution
            
            if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
                break;
            std::this_thread::sleep_for(0.1s);
        }

    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Temps d'exécution total : " << duration.count() << " millisecondes" << std::endl;
    MPI_Finalize();

    return EXIT_SUCCESS;
}
