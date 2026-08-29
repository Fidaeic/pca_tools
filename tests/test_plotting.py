from pca_tools import PCA


def test_score_plot_uses_one_interaction_parameter(sample_data):
    chart = PCA(n_comps=2).fit(sample_data).score_plot(1, 2)
    specification = chart.to_dict()

    assert len(specification.get("params", [])) == 1


def test_score_plot_rejects_a_component_outside_the_model(sample_data):
    model = PCA(n_comps=2).fit(sample_data)

    try:
        model.score_plot(1, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a component-range error")
