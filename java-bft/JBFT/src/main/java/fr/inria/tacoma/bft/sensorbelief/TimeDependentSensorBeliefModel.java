package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.mass.MassFunction;

/**
 * A time dependent sensor belief model is a model which uses an internal
 * memory to compute the mass function. It means that if you call this model
 * several times with the same arguments, you may not get the same result.
 */
public interface TimeDependentSensorBeliefModel<T> extends SensorBeliefModel<T> {

    /**
     * Maps a sensor value to a mass function with a manually specified elapsed
     * time since the last sensorMeasure. This function is only useful for models
     * which use temporization. For models where time does not change anything,
     * this will return the same mass function as toMass. Unlike
     * toMass(double), the real time will not be used and the function
     * will return a mass function as if elapsedTimeSeconds seconds have passed
     * since the last call.
     * @param sensorValue
     * @param elapsedTimeSeconds
     * @return
     */
    MassFunction toMass(T sensorValue, double elapsedTimeSeconds);


    /**
     * Create a mass function which correspond to the absence of sensor measure
     * with a manually specified elapsed since the last sensor measure.
     * If the model does not use previous sensor value, it will probably be a
     * vacuous mass function (all mass on full ignorance set). This is useful if
     * the model uses previous sensor value in order to get the current mass.
     * @return mass function if there are no sensor value
     */
    MassFunction toMassWithoutValue(double elapsedTimeSeconds);

}
