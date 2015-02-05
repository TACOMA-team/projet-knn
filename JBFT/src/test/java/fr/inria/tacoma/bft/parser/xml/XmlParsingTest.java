package fr.inria.tacoma.bft.parser.xml;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.sensorbelief.FusionSensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.LinearSensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.SpecificitySensorBeliefModel;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

public class XmlParsingTest {

    private Map<String, SensorBeliefModel> models;
    private FrameOfDiscernment frame;

    @Before
    public void setUp() throws Exception {
        models = XmlModelParser.parse(getClass().getResourceAsStream("/belief-model-1.xml"));
        frame = FrameOfDiscernment.newFrame("unittest", "A", "B", "C");
    }

    @Test
    public void XmlParser_parseTestConfig_returnConfigWith4Sensors(){
        assertEquals(models.size(), 4);
    }

    @Test
    public void XmlParser_parseTestConfig_ReturnConfigWithRightFrame() {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("unittest", "A", "B", "C");
        assertTrue("All models should match the test frame " + frame,
                models.values().stream().allMatch(model -> model.getFrame().equals(frame)));
    }

    @Test
    public void Model_forS1andS2_shouldBeASimpleLinearModel() {
        assertEquals(getExpectedModelForS1andS2(), models.get("S1"));
        assertEquals(getExpectedModelForS1andS2(), models.get("S2"));
    }

    @Test
    public void Model_forS3_ShouldBeTempoSpecificity() {
        SensorBeliefModel model = models.get("S3");
        assertTrue("S3 should have a model with specificity temporization",
                model instanceof SpecificitySensorBeliefModel);
        SpecificitySensorBeliefModel specificityModel = (SpecificitySensorBeliefModel) model;
        assertEquals(getExpectedUnderlyingModelForS3(), specificityModel.getUnderLyingModel());
    }

    @Test
    public void Model_forS3_ShouldBeTemporizationWithFusion() {
        SensorBeliefModel model = models.get("S4");
        assertTrue("S4 should have a model with fusion temporization",
                model instanceof FusionSensorBeliefModel);
        FusionSensorBeliefModel fusionModel = (FusionSensorBeliefModel) model;
        assertEquals(getExpectedUnderlyingModelForS4(), fusionModel.getUnderlyingModel());
    }


    private LinearSensorBeliefModel getExpectedModelForS1andS2() {
        LinearSensorBeliefModel expected = new LinearSensorBeliefModel(frame);
        expected.addInterpolationPoint(100.0, frame.toStateSet("B"), 0.75);
        expected.addInterpolationPoint(100.0, frame.toStateSet("C"), 0.25);

        expected.addInterpolationPoint(200.0, frame.toStateSet("A"), 0.25);
        expected.addInterpolationPoint(200.0, frame.toStateSet("B"), 0.5);
        expected.addInterpolationPoint(200.0, frame.toStateSet("C"), 0.1);
        expected.addInterpolationPoint(200.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(300.0, frame.toStateSet("A"), 0.75);
        expected.addInterpolationPoint(300.0, frame.toStateSet("B"), 0.1);
        expected.addInterpolationPoint(300.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(400.0, frame.toStateSet("A"), 0.25);
        expected.addInterpolationPoint(400.0, frame.toStateSet("B"), 0.5);
        expected.addInterpolationPoint(400.0, frame.toStateSet("C"), 0.1);
        expected.addInterpolationPoint(400.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(500.0, frame.toStateSet("B"), 0.75);
        expected.addInterpolationPoint(500.0, frame.toStateSet("C"), 0.25);

        return expected;
    }

    private LinearSensorBeliefModel getExpectedUnderlyingModelForS3() {
        LinearSensorBeliefModel expected = new LinearSensorBeliefModel(frame);
        expected.addInterpolationPoint(100.0, frame.toStateSet("B"), 0.75);
        expected.addInterpolationPoint(100.0, frame.toStateSet("C"), 0.25);

        expected.addInterpolationPoint(200.0, frame.toStateSet("A"), 0.25);
        expected.addInterpolationPoint(200.0, frame.toStateSet("B"), 0.5);
        expected.addInterpolationPoint(200.0, frame.toStateSet("C"), 0.1);
        expected.addInterpolationPoint(200.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(300.0, frame.toStateSet("A"), 0.75);
        expected.addInterpolationPoint(300.0, frame.toStateSet("B"), 0.1);
        expected.addInterpolationPoint(300.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(400.0, frame.toStateSet("A"), 0.25);
        expected.addInterpolationPoint(400.0, frame.toStateSet("B"), 0.5);
        expected.addInterpolationPoint(400.0, frame.toStateSet("C"), 0.1);
        expected.addInterpolationPoint(400.0, frame.toStateSet("A", "B"), 0.15);

        expected.addInterpolationPoint(500.0, frame.toStateSet("B"), 0.75);
        expected.addInterpolationPoint(500.0, frame.toStateSet("C"), 0.25);

        return expected;
    }

    private LinearSensorBeliefModel getExpectedUnderlyingModelForS4() {
        return getExpectedUnderlyingModelForS3();
    }

}
