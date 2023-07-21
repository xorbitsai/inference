import React from "react";

class ModelCard extends React.Component {
  constructor(props) {
    super(props);
    this.launchModel = this.launchModel.bind(this);
  }

  launchModel() {
    fetch(this.props.postURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(this.props.jsonData),
    })
      .then((response) => {
        console.log(response);
        window.open(
          this.props.postURL.split("/v1/models")[0],
          "_blank",
          "noreferrer"
        );
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
