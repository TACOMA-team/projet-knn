package fr.inria.tacoma.knn.experimental.continuous;

import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;

import java.util.Map;

public interface ContinuousKnnBelief<T> extends SensorBeliefModel<T> {

    double getK();

    double getAlpha();

    Map<String, Double> getGammas();

    /**
     * create a copy of the model with the same parameters except for alpha
     * @param newAlpha new alpha to use in new belief model
     * @return a new belief model with the given alpha
     */
    ContinuousKnnBelief<T> withAlpha(double newAlpha);

    /**
     * create a copy of the model with the same parameters except for k
     * @param newK new k to use in new belief model
     * @return a new belief model with the given k
     */
    ContinuousKnnBelief<T> withK(double newK);

    ContinuousKnnBelief<T> withAlphaAndK(double newAlpha, double newK);
}
