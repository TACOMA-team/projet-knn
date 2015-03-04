package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.util.Mass;

/**
 * A model which keep the previous generated mass function and use it to
 * compute later conversion from sensor value to mass function.
 *
 * This model contains an underlying model, typically a LinearSensorBeliefModel.
 * When receiving a new sensor data, it uses this underlying model to create a
 * new mass function. The old mass function is weakened according to the time
 * elapsed since the last call to toMass. The model then compares the
 * specificity of both mass functions and the most specific is returned to the
 * user.
 *
 * This is a way of using old evidence in order to create a new belief. If the
 * new sensor value is not specific we can keep the old one (and discount the mass
 * function depending on the age of the evidence).
 */
public class SpecificitySensorBeliefModel<T> implements TimeDependentSensorBeliefModel<T> {

    private SensorBeliefModel<T> model;
    private MassFunction oldEvidence;
    private double oldTimeStamp = 0;
    private double maxTime;

    /**
     * Creates a model with temporization using specificity. It need an underlying
     * model and a maxTime in seconds. This time is the duration of an evidence.
     * After maxTime seconds, the old evidence will be equal to a mass function
     * with only the full ignorance set as focal point.
     * @param model underlying model used to compute new mass functions
     * @param maxTime duration for a mass function
     */
    public SpecificitySensorBeliefModel(SensorBeliefModel<T> model, double maxTime) {
        this.model = model;
        this.maxTime = maxTime;
    }

    /**
     * @return the underlying model used by the SpecificitySensorBeliefModel
     */
    public SensorBeliefModel getUnderLyingModel() {
        return model;
    }

    @Override
    public MassFunction toMass(T sensorValue) {
        double newTimeStamp = (double)System.nanoTime() / 1e9;
        if(oldEvidence == null) {
            this.oldTimeStamp = newTimeStamp;
        }
        return toMass(sensorValue, newTimeStamp - this.oldTimeStamp);
    }

    @Override
    public MassFunction toMass(T sensorValue, double elapsedTime) {
        MassFunction newValue = this.model.toMass(sensorValue);

        if(this.oldEvidence == null) {//initialization of the model as this is the first call
            this.oldEvidence = newValue;
            return new MassFunctionImpl(newValue);
        }

        MassFunction weakened = new MassFunctionImpl(this.oldEvidence);
        weakened.discount(elapsedTime / this.maxTime);

        if(Mass.specificity(newValue) >= Mass.specificity(weakened)) {
            this.oldEvidence = newValue;
            this.oldTimeStamp = this.oldTimeStamp + elapsedTime;
            return newValue;
        }

        return weakened;
    }

    @Override
    public MassFunction toMassWithoutValue() {
        double newTimeStamp = (double)System.nanoTime() / 1e9;
        if(oldEvidence == null) {
            this.oldTimeStamp = newTimeStamp;
        }
        return toMassWithoutValue(newTimeStamp - this.oldTimeStamp);
    }

    @Override
    public MassFunction toMassWithoutValue(double elapsedTimeSeconds) {
        if(this.oldEvidence == null) { //we have no previous evidence to rely on
            MassFunction massFunction = new MassFunctionImpl(this.getFrame());
            massFunction.putRemainingOnIgnorance();
            return massFunction;
        }
        MassFunction weakened = new MassFunctionImpl(this.oldEvidence);
        weakened.discount(elapsedTimeSeconds / this.maxTime);
        return weakened;
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return this.model.getFrame();
    }
}
