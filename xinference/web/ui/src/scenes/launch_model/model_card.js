import React from "react";
import { v1 as uuidv1 } from "uuid";

class ModelCard extends React.Component {
  constructor(props) {
    super(props);
    this.launchModel = this.launchModel.bind(this);
  }

  launchModel() {
    const uuid = uuidv1();
    const jsonDataWithID = {
      ...this.props.jsonData,
      model_uid: uuid,
      endpoint: this.props.url,
    };

    console.log("Sending request to: " + jsonDataWithID.endpoint);

    // First fetch request to initiate the model
    fetch(this.props.url + "/v1/models", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(jsonDataWithID),
    })
      .then((response) => {
        console.log(response);

        // Second fetch request to build the gradio page
        return fetch(this.props.url + "/v1/gradio/" + uuid, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(jsonDataWithID),
        });
      })
      .then((response) => {
        console.log(response);
        window.open(this.props.url + "/" + uuid, "_blank", "noreferrer");
      })
      .catch((error) => console.error("Error:", error));
  }

  render() {
    const styles = {
      card: {
        width: "280px",
        border: "1px solid #ddd",
        borderRadius: "20px",
        padding: "15px",
        boxShadow: "0 4px 8px 0",
      },
      img: {
        display: "block",
        margin: "0 auto",
        width: "200px",
        height: "180px",
        objectFit: "cover",
        borderRadius: "10px",
      },
      h2: {
        margin: "10px 10px",
        fontSize: "20px",
      },
      p: {
        fontSize: "14px",
        padding: "0px 0px 15px 0px",
      },
      button: {
        backgroundColor: "#4CAF50",
        color: "white",
        display: "block",
        padding: "10px 24px",
        margin: "0 auto",
        border: "none",
        borderRadius: "5px",
        cursor: "pointer",
      },
    };

    return (
      <div style={styles.card}>
        <img
          style={styles.img}
          src={this.props.imgURL}
          alt={this.props.serviceName}
        />
        <h2 style={styles.h2}>{this.props.serviceName}</h2>
        <p style={styles.p}>{this.props.description}</p>
        <button style={styles.button} onClick={this.launchModel}>
          Launch
        </button>
      </div>
    );
  }
}

export default ModelCard;
