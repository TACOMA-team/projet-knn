package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Simple sensor model which generates mass function from sensor values
 * (1 dimension) with a linear interpolation between mass functions.
 *
 * This model maps a sensor value to a mass function. A call to toMass
 * will create a new mass function which will be a linear interpolation of the
 * focal points in the model.
 *
 * For instance, if the model maps 100 to the mass function {{A}-> 0.5, {A,B} -> 0.5}
 * and 200 to {{A}-> 0.3, {A,B} -> 0.7}, calling toMass(150) will return
 * {{A}-> 0.4, {A,B} -> 0.6}.
 */
public class LinearSensorBeliefModel implements SensorBeliefModel<Double> {

    private final FrameOfDiscernment frame;
    private NavigableMap<Double, MassFunction> interpolationPoints;

    public LinearSensorBeliefModel(FrameOfDiscernment frame) {
        this.frame = frame;
        this.interpolationPoints = new TreeMap<>();
    }

    /*
     * model building
     */
    public void addInterpolationPoint(double sensorValue, StateSet set, double beliefValue) {
        MassFunction massFunction = this.interpolationPoints.get(sensorValue);
        if(null == massFunction) {
            massFunction = new MassFunctionImpl(this.frame);
            this.interpolationPoints.put(sensorValue, massFunction);
        }
        massFunction.set(set, beliefValue);
    }

    public void addInterpolationFunction(double sensorValue, MassFunction massFunction) {
        MassFunction massCopy = new MassFunctionImpl(massFunction);
        this.interpolationPoints.put(sensorValue, massCopy);
    }
    /*
     *
     */

    @Override
    public MassFunction toMass(Double sensorValue) {
        Map.Entry<Double, MassFunction> point1 = this.interpolationPoints.floorEntry(sensorValue);
        Map.Entry<Double, MassFunction> point2 = this.interpolationPoints.ceilingEntry(sensorValue);
        if(null == point1) {
            /* The sensor value is before the first interpolation point.*/
            return point2.getValue();
        }
        else if(null == point2) {
            /* The sensor value is after the last interpolation point. */
            return point1.getValue();
        }
        else if(point1.getKey() == point2.getKey()) {
            /* We got exactly the key as the sensor value*/
            return point1.getValue();

        }

        double ratio = (sensorValue - point1.getKey()) / (point2.getKey() - point1.getKey());
        return interpolation(ratio, point1.getValue(), point2.getValue());
    }

    @Override
    public MassFunction toMassWithoutValue() {
        MassFunction vacuousMass = new MassFunctionImpl(this.frame);
        vacuousMass.putRemainingOnIgnorance();
        return vacuousMass;
    }

    @Override
    public FrameOfDiscernment getFrame() {
        return this.frame;
    }

    private MassFunction interpolation(double ratio, MassFunction massFunction1, MassFunction massFunction2) {
        MassFunction result = new MassFunctionImpl(this.frame);
        massFunction1.foreachFocalElement((set, value) -> result.addToFocal(set, value * (1 - ratio)));
        massFunction2.foreachFocalElement((set, value) -> result.addToFocal(set, value * ratio));
        return result;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        LinearSensorBeliefModel that = (LinearSensorBeliefModel) o;
        return frame.equals(that.frame) && interpolationPoints.equals(that.interpolationPoints);
    }

    @Override
    public int hashCode() {
        int result = frame != null ? frame.hashCode() : 0;
        result = 31 * result + (interpolationPoints != null ? interpolationPoints.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return "LinearSensorBeliefModel{" + interpolationPoints + '}';
    }
}
