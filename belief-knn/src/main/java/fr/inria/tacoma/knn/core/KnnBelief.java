package fr.inria.tacoma.knn.core;

import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;

public interface KnnBelief<T> extends SensorBeliefModel<T> {
    int getK();

    double getAlpha();

    /**
     * create a copy of the model with the same parameters except for alpha
     * @param newAlpha new alpha to use in new belief model
     * @return a new belief model with the given alpha
     */
    KnnBelief<T> withAlpha(double newAlpha);

    /**
     * create a copy of the model with the same parameters except for k
     * @param newK new k to use in new belief model
     * @return a new belief model with the given k
     */
    KnnBelief<T> withK(int newK);

    KnnBelief<T> withAlphaAndK(double newAlpha, int newK);
}
