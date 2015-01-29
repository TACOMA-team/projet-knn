package fr.inria.tacoma.bft.sensorbelief;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SpecificityModelTest {

    private FrameOfDiscernment frame;
    private LinearSensorBeliefModel baseModel;
    private SpecificitySensorBeliefModel specificityModel;
    private double evidenceDuration;

    @Before
    public void setUp() throws Exception {
        frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");
        baseModel = new LinearSensorBeliefModel(frame);
        baseModel.addInterpolationPoint(100.0, frame.toStateSet("sitting"), 1.0);
        baseModel.addInterpolationPoint(200.0, frame.toStateSet("sitting", "standing"), 1.0);
        baseModel.addInterpolationPoint(300.0,
                frame.toStateSet("sitting", "standing", "layingDown"), 1.0);
        evidenceDuration = 1.0;
        specificityModel = new SpecificitySensorBeliefModel(baseModel, evidenceDuration);
    }


    @Test
    public void sensorModelWithTemporization_firstCalltoMassFunction_ReturnsTheSameAsUnderlyingModel() {
        assertEquals(specificityModel.toMass(100), specificityModel.toMass(100, 0));
    }

    @Test
    public void sensorModelWithTemporization_newEvidenceMoreSpecific_ReturnsTheSameAsUnderlyingModel() {
        specificityModel.toMass(100, 0);//first call
        assertEquals(specificityModel.toMass(100), specificityModel.toMass(100, 0.5));
    }

    @Test
    public void sensorModelWithTemporization_newEvidenceLessSpecific_ReturnsTheSameAsUnderlyingModel() {
        specificityModel.toMass(150, 0);//first call
        MassFunction expected = new MassFunctionImpl(frame);
        expected.addToFocal(frame.toStateSet("sitting"), 0.5);
        expected.addToFocal(frame.toStateSet("sitting", "standing"), 0.5);
        expected.discount(0.5);
        assertEquals(expected, specificityModel.toMass(300, 0.5));
    }

    @Test
    public void toMassFunction_WithoutSensorMeasureAsFirstCall_ReturnsVacuousMassFunction() {
        MassFunction vacuousMass = new MassFunctionImpl(this.frame);
        vacuousMass.putRemainingOnIgnorance();
        assertEquals(vacuousMass, specificityModel.toMassWithoutValue());
    }

    @Test
    public void toMassFunction_WithoutSensorMeasureAsSecond_ReturnsWeakeanedPreviousFunction() {
        MassFunction firstMass = this.specificityModel.toMass(200.0);
        firstMass.discount(0.5);
        assertEquals(firstMass, this.specificityModel.toMassWithoutValue(
                evidenceDuration / 2));
    }
}
