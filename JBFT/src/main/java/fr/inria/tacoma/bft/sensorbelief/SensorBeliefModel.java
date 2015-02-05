package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;

/**
 * SensorBeliefModel is a mapping between sensor values and mass function.
 */
public interface SensorBeliefModel<T> {
    /**
     * Maps a sensor value to a mass function. If the model uses temporization
     * (is time dependant) then the real time will be used.
     * @param sensorValue value  to map to a mass function.
     * @return a mass function corresponding to the sensor value
     */
    MassFunction toMass(T sensorValue);

    /**
     * Create a mass function which correspond to the absence of sensor measure.
     * If the model does not use previous sensor value, it will probably be a
     * vacuous mass function (all mass on full ignorance set). This is useful if
     * the model uses previous sensor value in order to get the current mass.
     * @return mass function if there are no sensor value
     */
    MassFunction toMassWithoutValue();

    /**
     * @return The frame of discernment used by this model.
     */
    FrameOfDiscernment getFrame();
}
