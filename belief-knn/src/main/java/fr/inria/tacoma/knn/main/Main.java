package fr.inria.tacoma.knn.main;

import fr.inria.tacoma.knn.unidimensional.Main1D;
import fr.inria.tacoma.knn.bidimensional.Main2D;

import javax.swing.*;
import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Main {

    // Déclaration des fichiers :
    public static final String PRESENCE = "pres1a";
    public static final String ABSENCE = "abs1a";
    public static final String PRESENCE_TEST = "pres1t";
    public static final String ABSENCE_TEST = "abs1t";

    // Déclaration du nombre de variables :
    public static final int NB_OPTION = 5; // Nombre de variables possibles
    public static final int NB_FUNC = 2; // Nombre de fonctions de distance

    // Choix du mode :
    public static final int MODE = 2; // 1 : Montre la courbe pour ALPHA,K
                                      // 2 : Ecrit dans le fichier "Result.txt" l'erreur en fonction de alpha,k
                                      // 3 : Cherche les meilleures k et alpha pour toutes les features
                                      // 4 : Cherche les meilleures k et alpha pour une feature avec le gradient
                                      // 5 : Cherche les meilleures k et alpha pour toutes les features avec le gradient
    

    // Choix des variables (Modes 1 et 2) :
    public static final int DIM = 2; // Dimension 1 ou 2
    public static final int OPTION = 1; // 0 : centred val ; 1 : derivation ; 2 : moyenne ; 3 : max ; 4 : min
    public static final int OPTION2 = 3; // 2ème dimension
    public static final int FUNC = 0; // 0 : Décroissance exponentielle ; 1 : constante

    // Choix Variables (Mode 1) :
    public static final double ALPHA = 0.98;
    public static final int K = 6;

    // Choix du pas (Modes 2 et 3) :
    public static final double ALPHA_STEP = 0.05;
    public static final int K_STEP = 1;
    public static final int K_MAX = 20;

    // Coefficients du gradients (Mode 4) :
    public static final double GAMMA1 = 0.1;
    public static final double GAMMA2 = 0.001;

    // Variables chiantes :
    public static final int SENSOR_VALUE_CENTER = 512;
    public static final int MAX_X = 1024;
    public static final int MAX_Y = 1024;

    // Lance la bonne fonction en fonction du mode et de la dimension choisie (voir description des modes
    public static void main(String[] args) throws IOException {
        if (MODE == 3)
            TestAll() ;
        else {
            if (MODE == 5)
                SuperSayen() ;
        else {
            if (DIM == 1) {
                Main1D Hati = new Main1D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, FUNC, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, MODE, OPTION, SENSOR_VALUE_CENTER, MAX_X) ;
                Hati.main1D();
            }
            if (DIM == 2) {
                Main2D Hati = new Main2D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, FUNC, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, MODE, OPTION, OPTION2, SENSOR_VALUE_CENTER, MAX_X, MAX_Y) ;
                Hati.main2D();
            }
        }}
    }

    // Cherche le meilleur k et alpha pour toutes les features et toutes les fonctions de distance possibles
    public static void TestAll() throws IOException {
        for (int opt = 0 ; opt < NB_OPTION ; opt++){
            for (int func = 0 ; func < NB_FUNC ; func++){
                Main1D Hati = new Main1D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, func, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, MODE, opt, SENSOR_VALUE_CENTER, MAX_X) ;
                Hati.main1D();
            }
        }
        for (int opt = 0 ; opt < NB_OPTION ; opt++){
            for (int opt2 = opt + 1 ; opt2 < NB_OPTION ; opt2++){
                for (int func = 0 ; func < NB_FUNC ; func++){
                    Main2D Hati = new Main2D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, func, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, MODE, opt, opt2, SENSOR_VALUE_CENTER, MAX_X, MAX_Y) ;
                    Hati.main2D();
                }
            }
        }
    }
    
    // Cherche le meilleur k et alpha pour toutes les features et toutes les fonctions de distance possibles avec le gradient
    public static void SuperSayen() throws IOException {
        for (int opt = 0 ; opt < NB_OPTION ; opt++){
            for (int func = 0 ; func < NB_FUNC ; func++){
                Main1D Hati = new Main1D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, func, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, 4, opt, SENSOR_VALUE_CENTER, MAX_X) ;
        System.out.println("Dimension = 1 ; Option = " + opt + " ; Fontion = " + func + " ->\n    ") ;
                Hati.main1D();
            }
        }
        for (int opt = 0 ; opt < NB_OPTION ; opt++){
            for (int opt2 = opt + 1 ; opt2 < NB_OPTION ; opt2++){
                for (int func = 0 ; func < NB_FUNC ; func++){
                    Main2D Hati = new Main2D(PRESENCE, ABSENCE, PRESENCE_TEST, ABSENCE_TEST, ALPHA, K, func, ALPHA_STEP, K_STEP, K_MAX, GAMMA1, GAMMA2, 4, opt, opt2, SENSOR_VALUE_CENTER, MAX_X, MAX_Y) ;
        System.out.println("Dimension = 2 ; Option = " + opt + ", " + opt2 + " ; Fontion = " + func + " ->") ;
                    Hati.main2D();
        System.out.println(" ") ;
                }
            }
        }
    }
}
